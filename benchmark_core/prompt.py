from benchmark_core.config import Config
import inspect
import json
import os
from typing import TYPE_CHECKING, Any
from benchmark_core.data_process import safe_json_loads, extract_last_score_part, extract_prefix  # 文件顶部集中导入一次

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - allows prompt packing without transformers installed
    AutoTokenizer = None

if TYPE_CHECKING:
    from runner import VLLMRunner
else:  # pragma: no cover - runtime fallback for environments without runner deps
    VLLMRunner = Any

def _manual_chat_prompt(system: str, user: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    }


def _signature_accepts_kwargs(callable_obj: Any) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _configured_chat_template_kwargs(tokenizer: Any) -> dict[str, Any]:
    configured = Config.get("generation_chat_template_kwargs") or {}
    if not isinstance(configured, dict) or not hasattr(tokenizer, "apply_chat_template"):
        return {}
    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_kwargs = _signature_accepts_kwargs(tokenizer.apply_chat_template)
    return {
        key: value
        for key, value in configured.items()
        if accepts_kwargs or key in sig.parameters
    }


def _filter_chat_template_base_kwargs(tokenizer: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_kwargs = _signature_accepts_kwargs(tokenizer.apply_chat_template)
    filtered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in sig.parameters:
            filtered[key] = value
        elif accepts_kwargs and key not in {"add_generation_prompt"}:
            filtered[key] = value
    return filtered


def _apply_chat_template_kwargs(tokenizer: Any, base_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(_configured_chat_template_kwargs(tokenizer))
    kwargs.update(_filter_chat_template_base_kwargs(tokenizer, base_kwargs))
    return kwargs


def _normalize_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(message) for message in messages]
    if (
        Config.get("generation_chat_template_no_system_role")
        and len(normalized) >= 2
        and normalized[0].get("role") == "system"
        and normalized[1].get("role") == "user"
    ):
        system_text = str(normalized[0].get("content") or "")
        user_text = str(normalized[1].get("content") or "")
        merged_user = dict(normalized[1])
        merged_user["content"] = f"{system_text}\n\n{user_text}".strip()
        normalized = [merged_user] + normalized[2:]
    return normalized


class GeneratePromptFormatter:
    """
    Only generation prompts need tokenizer-aligned chat formatting.
    All other prompt classes should return manual system+user messages.
    """

    def __init__(self, model: Any):
        self.model_name = getattr(model, "model_name", "")
        self.tokenizer = None
        if AutoTokenizer is not None and self.model_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    trust_remote_code=True,
                )
            except Exception:
                self.tokenizer = None

    def render(self, messages: list[dict[str, Any]]) -> Any:
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            return messages

        try:
            messages = _normalize_chat_template_messages(messages)
            kwargs: dict[str, Any] = {"tokenize": False}
            sig = inspect.signature(self.tokenizer.apply_chat_template)
            has_prefill = bool(messages) and messages[-1].get("role") == "assistant" and messages[-1].get("prefix")
            supports_continue = "continue_final_message" in sig.parameters or _signature_accepts_kwargs(self.tokenizer.apply_chat_template)
            if has_prefill and not supports_continue:
                raise ValueError(
                    f"Tokenizer for {self.model_name or 'current model'} does not support "
                    "continue_final_message, but Generate_Prompt requires assistant continuation."
                )
            if "add_generation_prompt" in sig.parameters:
                kwargs["add_generation_prompt"] = not has_prefill
            if supports_continue and has_prefill:
                kwargs["continue_final_message"] = True
            kwargs = _apply_chat_template_kwargs(self.tokenizer, kwargs)
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            raise

class Generate_Prompt:
    """
    Generation prompt builder.
    This is the only prompt path that keeps tokenizer-aligned chat formatting.
    """
    def __init__(self, model: Any, query: str = None):
        self.query = query or ""
        self.model = model
        self.prompt_formatter = GeneratePromptFormatter(model)
        self.system_message = (
            "You are a mathematician. Solve the problem."
            "## Style preferences (keep them light; do not change your underlying approach):"
                "- Treat `current_solution`/`ref` as correct established premises and build directly on them."
                "- Start immediately with the next logical derivation. Do not restate the problem or re-summarize what has already been established."
                "- Write as continuous mathematical prose (no section headers, no “Step 1/2/3”)."
                "- Avoid repeating the same conditions. If you must reference a prior premise, do it minimally (e.g., “from the previous inequality …”)."
        )
        self.current_solution = ""
        self.schema = None
        self.prompt = ""
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name, use_fast=True)


    def add_step(self, step: str):
        if step:
            self.current_solution += "\n" + step if self.current_solution else step
        #print(f"[DEBUG] Generation query: {self.current_solution}")


    def return_prompt(self) -> str:
        if self.current_solution:
            message = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"Solve the Problem:\n{self.query}"},
                {"role": "assistant", "content": self.current_solution, "prefix": True}
            ]
        else:
            message = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"Problem:\n{self.query}"}
            ]
        self.prompt = self.prompt_formatter.render(message)
        return self.prompt

    def run(self) -> str:
        prompt = self.return_prompt()
        out = self.model.generate(prompt, self.schema).strip()
        return out

class Pairwise_Prompt:
    def __init__(self, model: Any):
        self.model = model
        self.user_message = ""
        self.system_message = self.system_message = (
        """Reasoning: High.

        ## Role

        You are tasked with determining whether the **current reasoning step** (CURRENT_STEP) **violates or distorts** the logic, assumptions, or conclusions established in a single **dependency claim** (DEPENDENCY_CLAIM). This evaluation identifies contradictions, hallucinations, or inconsistencies between the current step and the claims it explicitly depends on.

        You are given three texts:

        - GLOBAL_PREFIX: The earlier reference steps that precede CURRENT_STEP. This is provided **for context only**, and should only be used to verify notation or local context when necessary. It should **not** be used as an additional source of constraints.

        - DEPENDENCY_CLAIM: One claim that the current step depends on. This serves as your **local anchor** for comparison.

        - CURRENT_STEP: The current generated step that you must evaluate for consistency against DEPENDENCY_CLAIM.

        ---

        ## Key Responsibilities

        1. **Your primary task is to evaluate CURRENT_STEP** in relation to DEPENDENCY_CLAIM. Specifically, determine whether CURRENT_STEP introduces contradictions, alters conditions without justification, or misinterprets symbols and conclusions.

        2. **DEPENDENCY_CLAIM is your normative reference**. All judgments must be made **solely** by comparing CURRENT_STEP to DEPENDENCY_CLAIM. This is your local contract.

        3. **GLOBAL_PREFIX is read-only background**. You may reference GLOBAL_PREFIX **only** to **verify** that any perceived contradiction is not due to a lack of context in DEPENDENCY_CLAIM and CURRENT_STEP alone. If the perceived conflict can be explained or resolved by GLOBAL_PREFIX, do not treat it as an inconsistency.

        ---

        ## Hard Information Constraints

        You must adhere to the following strict rules:

        1. **Only use the information** explicitly present in:
        - DEPENDENCY_CLAIM
        - CURRENT_STEP
        - Optionally, you may use GLOBAL_PREFIX to verify whether the perceived conflict between CURRENT_STEP and DEPENDENCY_CLAIM is due to incomplete information (but **not** to judge directly).

        2. **You must ignore** any of the following:
        - Any steps that occur after the dependency claim, even if they appear in GLOBAL_PREFIX.
        - The original textual problem statement (unless it is explicitly included in DEPENDENCY_CLAIM).
        - Any background mathematical knowledge that is not explicitly derived from DEPENDENCY_CLAIM.

        3. **You may use basic logical and algebraic reasoning**, but **only** for the specific expressions, variables, and conditions appearing in DEPENDENCY_CLAIM and CURRENT_STEP.

        4. If a contradiction is suspected between DEPENDENCY_CLAIM and CURRENT_STEP, **first check whether this contradiction can be resolved by GLOBAL_PREFIX**. If GLOBAL_PREFIX explains or justifies the conflict, **treat CURRENT_STEP as consistent**.

        5. If you are unsure whether a true contradiction exists, you must **err on the side of consistency** and assign CURRENT_STEP the **higher score**.

        ---

        ## Types of Inconsistencies / Hallucinations (Relative to DEPENDENCY_CLAIM)

        CURRENT_STEP is considered **inconsistent** or hallucinatory if one or more of the following occurs:

        ### 1. Direct Logical Conflict
        - **CURRENT_STEP reverses, negates, or alters a condition** stated in DEPENDENCY_CLAIM without any valid explanation or justification.

        ### 2. Symbol or Context Inconsistency
        - A symbol or condition used in both DEPENDENCY_CLAIM and CURRENT_STEP is given **inconsistent meanings**.

        - CURRENT_STEP introduces assumptions that **contradict** the conditions in DEPENDENCY_CLAIM.

        ### 3. Ignoring Explicit Constraints
        - DEPENDENCY_CLAIM gives a crucial condition or assumption, and CURRENT_STEP **proceeds as if this condition does not exist**, making logical conclusions or claims incompatible with it.

        ### 4. **Scope Distortion (Weakening or Strengthening Conditions)**
        - CURRENT_STEP **weakens** a condition or conclusion established in DEPENDENCY_CLAIM. This can lead to an erroneous generalization or misunderstanding of the scope.
        
        - CURRENT_STEP **strengthens** a condition without proper justification.

        - **Any distortion of scope** (either weakening or strengthening) must be clearly explained or justified by CURRENT_STEP. If not, it is considered inconsistent.

        ---

        ## What IS Allowed

        CURRENT_STEP **can**:

        - Introduce **subcases or auxiliary reasoning constructs** that do not contradict or weaken any part of DEPENDENCY_CLAIM.

        - Introduce **new assumptions or definitions**, provided they do not contradict any explicitly stated or implied assumption in DEPENDENCY_CLAIM.

        - Extend the reasoning in a way that **preserves the validity** of the original conclusions in DEPENDENCY_CLAIM.

        ---

        ## Scoring Rules (0–5)

        Your judgment reflects **how severely CURRENT_STEP violates** the consistency and logic of DEPENDENCY_CLAIM, not whether it is globally incorrect.

        - **5 – Fully Consistent:** CURRENT_STEP fully respects and extends the reasoning of DEPENDENCY_CLAIM.

        - **4 – Minor Issues:** CURRENT_STEP introduces small ambiguities or unusual inferences, but these do not conflict directly with DEPENDENCY_CLAIM.

        - **3 – Weak Consistency:** CURRENT_STEP mostly aligns with DEPENDENCY_CLAIM but may be poorly argued or introduce some unclear logic.

        - **2 – Significant Conflict:** CURRENT_STEP clearly contradicts or omits important elements from DEPENDENCY_CLAIM.

        - **1 – Severe Misalignment:** CURRENT_STEP largely ignores or misuses the conditions or conclusions established in DEPENDENCY_CLAIM.

        - **0 – Direct Contradiction:** CURRENT_STEP fundamentally contradicts DEPENDENCY_CLAIM. The contradiction is irreconcilable and cannot be resolved by GLOBAL_PREFIX.

        When you are truly unsure between two adjacent scores, you must always choose the **higher score** to ensure the most favorable judgment.

        ---

        ## Output Format (Strict)

        Return **only** a JSON object of the following form:

        `{"score": k}`

        Where `k` is an integer from `{0,1,2,3,4,5}`.  

        - No explanations, comments, or natural language text.  
        - No additional keys like `"analysis"`, `"score"`, or `"tags"`.  
        - The entire reply must **only** contain one **JSON object**.

        """)


        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [0, 1, 2, 3, 4, 5],
                    "description": "discrete consistency score (0–5), higher = more consistent"
                }
            },
            "required": ["score"],
            "additionalProperties": False
        }
        # 新增：缓存一份全局前缀，方便多次复用
        self.global_prefix = ""

    def set_global_prefix(self, prefix: str | None):
        """在一轮实验开始前，先把整段前缀传进来缓存一下。"""
        self.global_prefix = prefix or ""

    def build_user(self, gen_text: str, ref_text: str, prefix: str | None = None) -> None:
        """
        prefix:
            - 如果传入，则覆盖 self.global_prefix
            - 如果为 None，则使用之前 set_global_prefix 的缓存
        """
        if prefix is None:
            prefix = self.global_prefix

        self.user_message = (
            "## GLOBAL_PREFIX (background only; DO NOT use it as a source of constraints)\n"
            f"{prefix}\n\n"
            "## Task\n"
            "- Perform a **pairwise contradiction / hallucination check** between the current step **CURRENT_STEP** and the single dependency claim **DEPENDENCY_CLAIM** that this step depends on.\n"
            "- Treat **DEPENDENCY_CLAIM as your only normative local context for judging inconsistency**.\n"
            "- You may read GLOBAL_PREFIX only to understand notation, but you MUST NOT use it as extra evidence of conflict.\n"
            "- When uncertain whether there is a conflict based on DEPENDENCY_CLAIM alone, treat CURRENT_STEP as **consistent** and choose the **higher score**.\n"
            "\n"
            "## DEPENDENCY_CLAIM (local dependency-claim anchor for CURRENT_STEP)\n"
            f"{ref_text}\n\n"
            "## CURRENT_STEP\n"
            f"{gen_text}\n\n"
            "## Output Requirements\n"
            "- Output a single strict JSON object: `{{\"score\": k}}` where `k ∈ {{0,1,2,3,4,5}}`.\n"
            "- Valid outputs: `0, 1, 2, 3, 4, 5`.\n"
        )

    def return_prompt(self) -> str:
        sys = self.system_message + "\n\nReturn only a valid json object, e.g. {\"score\": 5}."
        usr = self.user_message + "\n\nRemember: output json only. Example: {\"score\": 3}."

        return {
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]
        }

    def run(self, gen_claim: str, ref: list[str], prefix: str | None = None) -> dict:
        """
        gen_claim: 当前要评估的 GEN（完整一步或前缀）
        ref:       多个 REF_STEP（依赖 claim 文本），逐个和 GEN 做 pairwise 检查
        prefix:    整段 GLOBAL_PREFIX（通常就是完整的已有解答前缀）
                    - 可以传入；如果为 None，则使用 self.global_prefix
        """
        prompts = []
        scores = []

        for ref_step in ref:
            # 这里所有 ref_step 共用同一个 prefix
            self.build_user(gen_claim, ref_step, prefix=prefix)
            prompt = self.return_prompt()
            prompts.append(prompt)

        reasonings, outs = self.model.generate(prompts, None)

        for out in outs:
            score = extract_last_score_part(out)
            scores.append(score)

        #print("pairwise scores:", scores)
        return {
            "scores": scores,
            "raw_outputs": outs,
            "reasoning_outputs": reasonings,
            "gen": gen_claim,
            "refs": ref,
        }


class Holistic_Prompt:
    def __init__(self, model: Any):
        self.model = model
        self.user_message = ""
        self.system_message = (
            """
            Reasoning: High.

            ## Role

            You are tasked with evaluating the **continuity of reasoning structure** in a step-by-step mathematical solution.

            Your job is **not to solve the entire problem**, but to assess whether the **next step (GEN)** logically follows and continues the reasoning structure established in the **reference (REF)**. This means focusing on the **logical flow** of the solution, not on specific mathematical details (such as arithmetic or algebraic calculations). Your goal is to determine if GEN **advances the reasoning structure** appropriately, according to the methods and approach established in REF.

            ### Key Concepts

            ### 1. Structural Consistency

            - **Does GEN continue to follow the same overall methodology or plan** as the reference? For example, if REF uses **mathematical induction**, does GEN continue using induction to prove the result? Does GEN still adhere to the same **case division**, **substitutions**, or **transformations** employed in REF?

            - **Does GEN respect the context established in REF**? What are the existing symbols, variables, and sub-goals introduced in the original text? Does GEN keep using the same symbols with the same meaning, and does it build upon what was previously established?

            - **Does GEN proceed naturally** from the reasoning developed in REF? Even if the reasoning is not strictly sequential, for example in **proof by contradiction** or **backward reasoning**, does GEN still make sense within the ongoing logical structure of REF?

            #### Special Cases:

            - **Backward Reasoning (e.g., conclusion-first reasoning)**: In some proofs, the conclusion is assumed first, and then the argument proceeds to show that it holds. This approach is common in **proofs by contradiction** and **reductio ad absurdum**. **GEN should not be penalized** for employing backward reasoning if REF follows this structure. Instead, GEN should continue by either proving the assumed conclusion or negating the assumption that leads to a contradiction.

            - **Non-linear Reasoning (e.g., proof by contradiction, indirect proof)**: If REF employs **proof by contradiction**, where an assumption is made and then a contradiction is derived, GEN should **continue this approach** without penalty. Ensure that GEN respects the **contradiction structure** set up in REF, even if the steps appear to proceed out of the expected order or appear indirect.

            - For instance, if REF assumes `A` to derive `B` and then shows that assuming `A` leads to a contradiction, GEN can conclude that `A` is false and thus `not A` is true, provided it follows the reasoning from REF.

            - **Case Analysis (Classification)**: When REF uses **case analysis** (dividing into different cases and treating each one separately), it is important that GEN **correctly respects this structure**. Each case in the analysis should be handled independently, and GEN should continue reasoning within the correct case without prematurely generalizing or skipping between cases.

            #### Special Note on Case Analysis:

            - **Case Analysis and Correct Handling of Cases**: Case analysis is a legitimate and often necessary tool in mathematical reasoning. When GEN proceeds through different cases, **it should not be penalized** simply for switching between cases or using different cases in separate steps. However, if GEN **inappropriately skips a case** or **misapplies a case’s logic** (e.g., drawing conclusions outside the scope of the case's assumption), this should be penalized.

            - **Example**: If REF analyzes two cases (Case 1: `x > 0`, Case 2: `x ≤ 0`), and GEN jumps from one case to the other without properly completing or justifying each case, this would be a **structural error**. GEN must **complete the logical process in each case** before moving to the next.

            ### 2. Local Logical Connections

            - Can a careful reader discern how GEN **naturally derives** from the last few steps in REF? Even if intermediate algebraic operations or detailed steps are skipped, does GEN logically progress from what was previously established in REF?

            - Does the reasoning appear to **advance logically** and make sense in the context of the ongoing proof or argument? The **logical connections** between REF and GEN should be clear, and GEN should make reasonable **progress** in solving or proving the problem.

            - **Example**: If REF has shown that a certain inequality holds for a case, GEN can proceed by applying this inequality in a different step or refining it further. This would be acceptable, even if the detailed arithmetic operations are omitted, as long as the logical connection remains strong.

            - Avoid penalizing **minor mathematical errors** or **small inconsistencies** that do not disrupt the logical flow of reasoning. If a minor mistake does not affect the overall logic or progression of the proof, the focus should remain on the **logical flow** rather than the specific computations.

            ### 3. Effective Continuation and Stagnation

            - Does GEN **advance** the reasoning? Does it introduce a **new step**, such as deriving a new inequality, simplifying an expression, or setting up a subproblem?

            - **Example**: If REF derived an intermediate result, GEN might use that result to derive a further step, such as a new inequality or condition, or it might simplify the expression derived in REF. This would demonstrate effective continuation.

            - Alternatively, does GEN **repeat vague comments** or **restate** steps that do not move the solution forward? If GEN simply reiterates previous ideas or steps without adding new value or logical progression, this should be **penalized**.

            - **Critical:** If GEN **jumps directly to the final answer** without continuing the **logical reasoning** established in REF, this should be penalized. If GEN skips over necessary intermediate steps or does not justify its conclusions through the ongoing logical process, this creates a **break in the reasoning structure**.

            - **Example**: If REF develops a complex argument over several steps and GEN suddenly jumps to the final conclusion without addressing the necessary intermediate steps, it could be seen as **skipping important reasoning**, which breaks the logical flow.

            ### 4. Structural Deviations and Dead Ends

            - Does GEN **switch to a completely different approach** without justification? For example, if REF employs a **geometric proof**, does GEN suddenly switch to **combinatorics** or another unrelated approach without an explanation?

            - **Example**: If REF proves a result using induction, but GEN introduces a new, unrelated method (such as geometric reasoning or a completely different assumption) to continue the proof without any justification, this should be marked as a **structural deviation**.

            - Does GEN **abandon the ongoing plan** and pursue a new, unrelated direction? For example, if REF is progressing through a direct proof but GEN suddenly introduces a contradiction without a clear connection to REF, this should be marked as a **disruption in reasoning**.

            - Does GEN **enter a dead end** in the reasoning? This could happen if GEN attempts to proceed with the proof but the path leads to an unsolvable or illogical conclusion due to the previous steps in REF. This could be because of invalid assumptions, contradiction, or failure to respect previously established conditions.

            - **Example**: If REF establishes that `x > 0`, and GEN later assumes `x ≤ 0` without addressing this contradiction, the reasoning is at a **dead end**.

            ---

            ## Scoring (0–5)

            Give an integer score `"score"`. `{0,1,2,3,4,5}`:

            - **5 – Excellent Structural Continuity**  
            GEN follows REF’s plan flawlessly. The logic is well-connected, and GEN makes **significant progress** in the proof, without introducing unnecessary deviations or dead ends.

            - **4 – Good Structural Continuity, but with Minor Issues**  
            GEN largely follows the reasoning in REF and is logically coherent, with **minor issues** (e.g., slight vagueness, minor deviations in the logical flow, or slightly rushed transitions). The overall structure remains intact.

            - **3 – Weak Structural Continuity, but Acceptable**  
            GEN maintains a connection to REF, but the reasoning is vague or unclear. The logical connections are not strong, and GEN adds limited value in terms of **advancing the solution**.

            - **2 – Questionable Structure**  
            GEN diverges noticeably from the reasoning in REF. The logical connections are **weak**, and GEN introduces issues like **logical gaps**, irrelevant steps, or unaddressed contradictions that make the reasoning unclear.

            - **1 – Poor Structural Continuity**  
            GEN significantly deviates from the reasoning established in REF. There are **major logical flaws**, contradictions, or **abandoned reasoning** that make GEN largely unrelated to REF's plan.

            - **0 – No Meaningful Structural Relationships**  
            GEN is **completely unrelated** to REF. It fails to build upon the reasoning established in REF, either by **contradicting the prior logic** or by introducing a **new approach** that is completely disconnected from the established reasoning structure.

            ---

            ## Output Format

            Your final output should be a **single JSON object** in the following format:

            `{"score": X}`

            Where `X` is an integer in the range `{0,1,2,3,4,5}`.

            **Do not** include any additional explanations, comments, or text in your final answer.  
            **Only output the JSON object.**
            """
        )
        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [0, 1, 2, 3, 4, 5],
                    "description": "discrete degree of entailment (0–5), higher = stronger entailment"
                }
            },
            "required": ["score"],
            "additionalProperties": False
        }
        
    def build_user(self, gen_text: str, ref_text: str) -> str:
        self.user_message = (
            "## Task\n"
            "Judge whether **GEN** faithfully **CONTINUES THE SAME METHOD / FLOW** committed in **REF** (all prior steps).\n"
            "Penalize **route switching**, **jumping ahead** (skipping moves implied by REF), or **breaking prior commitments**.\n"
            "Use **REF only**; no outside knowledge. When uncertain between two scores, **choose the lower**.\n"
            "\n"
            "## Inputs\n"
            "### REF (all prior steps up to now)\n"
            f"{ref_text}\n\n"
            "### GEN (the immediate next step)\n"
            f"{gen_text}\n\n"
            "## Output\n"
            "- Strict JSON: `{{\"score\": k}}` where `k ∈ {{0,1,2,3,4,5}}`.\n"
            "- Valid outputs: `0, 1, 2, 3, 4, 5`.\n"
        )
        
    def return_prompt(self) -> str:
        sys = self.system_message + "\n\nReturn only a valid json object, e.g. {\"score\": 5}."
        usr = self.user_message + "\n\nOutput json only. Example: {\"score\": 4}."
        return {
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]
        }
    
    def run(self, gen_claim: str, ref_claim: str) -> dict:
        """Returns a dict with the structural continuation score and raw model output."""
        self.build_user(gen_claim, ref_claim)
        prompt = self.return_prompt()
        reasonings, outs = self.model.generate([prompt], None)
        reasoning = reasonings[0] if reasonings else ""
        output = outs[0] if outs else ""
        score = extract_last_score_part(output)
        return {
            "score": score,
            "raw_output": output,
            "reasoning_output": reasoning,
            "gen": gen_claim,
            "ref": ref_claim,
        }

    
class SelfJudge_Prompt:
    def __init__(self, model: Any):
        self.model = model
        self.user_message = ""
        self.system_message = (
            """
            # Reasoning: High

            ## Role:
            You are an automated evaluator for **reference-free factual soundness** and **internal consistency**.

            ## Task:
            Without any REF, inspect **GEN** for internal mathematical correctness and self-consistency. Identify **arithmetic/algebraic mistakes**, **illegal operations**, **undefined or redefined symbols**, **incompatible constraints**, and **self-contradictions**.

            ## Rules:
            - **No outside knowledge**: Judge only by **logic/maths** that are explicitly stated or standardly valid given the expressions in GEN. You should only evaluate facts that are **explicitly available** in the GEN step. If the context is missing, assume that certain details are unavailable for evaluation, and **do not speculate**.
            
            - **Focus on immediate internal coherence**: Assess the **internal logic and math** in the GEN excerpt, such as:
            - **Variable definitions**
            - **Domain restrictions**
            - **Equation manipulations**
            - **Sign/inequality directions**
            - **Step-to-step consistency within GEN**
            
            - **Conservative approach for unverifiable claims**: If GEN introduces results or statements that are **unverifiable** due to missing context or the need for additional derivations, apply **caution**. Do not assume or extrapolate results beyond what is given in the GEN step.

            - **Penalize unverifiable claims**: If a result is presented without sufficient derivation, and derivation is needed within the GEN step to substantiate the result, **penalize appropriately**.

            - **Bias to caution when uncertain**: If uncertain about a step or claim due to incomplete context, **give the benefit of the doubt**, erring on the side of **caution**.

            ## Input:
            - **GEN**: a short mathematical reasoning excerpt.

            ## Output (strict):
            - **JSON only**: `{"score": k}` where `k` ∈ `{0,1,2,3,4,5}`; higher = fewer detectable internal errors/contradictions.

            ## Scoring Guide:
            - **5**: No detectable internal errors; operations and symbols are coherent and consistent.
            - **4**: Minor slips/omissions that do not change correctness.
            - **3**: Generally sound but with one or two questionable/under-justified links.
            - **2**: Clear error(s) in manipulation or conflicting constraints.
            - **1**: Multiple errors; reasoning largely unsound.
            - **0**: Nonsensical or self-contradictory throughout.

            ## Instruction:
            Evaluate GEN’s **internal mathematical correctness** and **self-consistency** only, ensuring that any information is **explicitly available** in the GEN step. Be cautious when context is incomplete, and avoid extrapolating information that isn't directly given. Output only `{"score": k}`.
        """)

        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [0, 1, 2, 3, 4, 5],
                    "description": "discrete degree of entailment (0–5), higher = stronger entailment"
                }
            },
            "required": [
                "score"
            ],
            "additionalProperties": False
        }
        
    def build_user_without_reference(self, gen_text: str) -> None:
        self.user_message = (
            "Task: Reference-free evaluation of GEN’s internal mathematical correctness and self-consistency.\n"
            "Check symbol definitions, domain/constraint compatibility, legality of operations, sign/inequality directions, "
            "and step-to-step coherence within GEN. Penalize unverifiable claims that require derivation inside GEN.\n"
            "Use no outside knowledge. When uncertain between two scores, choose the lower.\n"
            "Output strictly JSON: {{\"score\": k}} where k ∈ {{0,1,2,3,4,5}}.\n"
            "GEN:\n"
            f"{gen_text}\n"
            "Valid outputs: 0,1,2,3,4,5."
        )

    def build_user_with_reference(self, gen_text: str, ref_claim_text: str, step_label: str | None = None) -> None:
        label = step_label or "same reference step"
        self.user_message = (
            "Task: Evaluate the CURRENT_STEP against one claim from the same current step.\n"
            "Use the current-step claim as a local anchor for support, contradiction, and omission checking.\n"
            "Do not use claims from other steps. Do not use outside knowledge.\n"
            "If CURRENT_STEP conflicts with the current-step claim, lower the score. If CURRENT_STEP is merely under-justified, score conservatively.\n"
            "Output strictly JSON: {{\"score\": k}} where k ∈ {{0,1,2,3,4,5}}.\n"
            f"CURRENT_STEP_LABEL:\n{label}\n"
            f"CURRENT_STEP_CLAIM:\n{ref_claim_text}\n"
            f"CURRENT_STEP:\n{gen_text}\n"
            "Valid outputs: 0,1,2,3,4,5."
        )

        
    def return_prompt(self) -> str:
        return {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.user_message},
            ]
        }
    
    def run_without_reference(self, gen_claim: str) -> dict:
        self.build_user_without_reference(gen_claim)
        prompt = self.return_prompt()
        reasonings, outs = self.model.generate([prompt], None)
        output = outs[0] if outs else ""
        reasoning = reasonings[0] if reasonings else ""
        score = extract_last_score_part(output)
        return {
            "mode": "without_reference",
            "score": score,
            "raw_output": output,
            "reasoning_output": reasoning,
            "gen": gen_claim,
            "prompt": prompt,
        }

    def run_with_reference(self, gen_claim: str, ref_claims: list[str], step_label: str | None = None) -> dict:
        prompts = []
        for ref_claim in ref_claims:
            self.build_user_with_reference(gen_claim, ref_claim, step_label=step_label)
            prompts.append(self.return_prompt())

        if prompts:
            reasonings, outs = self.model.generate(prompts, None)
        else:
            reasonings, outs = [], []
        scores = [extract_last_score_part(item) for item in outs]
        score = sum(scores) / len(scores) if scores else -1
        return {
            "mode": "with_reference",
            "score": score,
            "scores": scores,
            "raw_output": outs,
            "reasoning_output": reasonings,
            "gen": gen_claim,
            "refs": ref_claims,
            "step_label": step_label,
            "prompts": prompts,
        }

    def run(self, gen_claim: str) -> dict:
        return self.run_without_reference(gen_claim)


class Claim_Segment_Prompt:
    def __init__(self, model: VLLMRunner):
        self.user_message = ""
        self.system_message = (
        "You are an mathemaical expert in natural language understanding. "
        "Task: Given a sentence or paragraph, segment it into "
        "atomic propositions (minimal semantic units that cannot be "
        "further decomposed without losing meaning). "
        "Follow these rules:\n"
        "1. Each atomic proposition must be a clear, independent statement.\n"
        "2. Do not paraphrase; preserve original meaning as much as possible.\n"
        "3. Number the propositions in order of appearance.\n"
        "4. Output STRICT JSON only, with the following format (≤10 items; each ≤80 chars):\n"
        "{"
        "\"segments\": [\n"
        "  {\"id\": <int>, \"text\": \"<atomic proposition>\"},\n"
        "  ...\n"
        "]\n"
        "}\n"
        "Ensure JSON is valid and contains no extra commentary."
        )
        self.model = model
        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "id": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "index number"
                    },
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 80,
                        "description": "atomic proposition (≤80 chars; complete statement)"
                    }
                    },
                    "required": ["id", "text"],
                    "additionalProperties": False
                },
                "minItems": 1
                }
            },
            "required": ["segments"],
            "additionalProperties": False
        }

        
    def build_user(self, text: str) -> str:
        self.user_message = (
            f"Segment the following text into atomic propositions:\n{text}\n"        
        )
    def return_prompt(self) -> str:
        sys = self.system_message
        usr = self.user_message + "\n\nRemember: output json only. Example: {\"segments\": [{\"id\": 0, \"text\": \"Proposition 1.\"}, {\"id\": 1, \"text\": \"Proposition 2.\"}]}"
        return {
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]
        }
    def run(self, text: str) -> dict:
        """返回严格 JSON：{"segments": [{"id": int, "text": str}, ...]}"""
        self.build_user(text)
        prompt = self.return_prompt()
        out = self.model.generate([prompt], self.output_schema)

        # Compatible with different runners:
        # - tuple(reasonings, contents)
        # - list[str]
        # - str
        # - already-decoded dict
        payload = out
        if isinstance(out, tuple) and len(out) >= 2:
            payload = out[1]
        if isinstance(payload, list):
            payload = payload[0] if payload else "{}"
        if isinstance(payload, dict):
            return payload
        return json.loads(payload)
