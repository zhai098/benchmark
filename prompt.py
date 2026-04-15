from config import Config
import json
import os
from typing import TYPE_CHECKING, Any
from data_process import safe_json_loads, extract_last_score_part, extract_prefix  # 文件顶部集中导入一次

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - allows prompt packing without transformers installed
    AutoTokenizer = None

if TYPE_CHECKING:
    from runner import VLLMRunner
else:  # pragma: no cover - runtime fallback for environments without runner deps
    VLLMRunner = Any

class PromptBuilder:
    def __init__(self, model: Any):
        self.model_name = model.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True) if AutoTokenizer else None

    def make_chat_prompt(self, system: str, user: str, add_generation_prompt=True, continue_final_message=False,
):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        if self.tokenizer is not None and hasattr(self.tokenizer, 'chat_template'):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, 
                                                      add_generation_prompt=add_generation_prompt,
                                                      continue_final_message=continue_final_message)
            return text
        else:
            # 如果不支持聊天模板，使用简单的提示构建方法
            return '\n'.join([f"{message['role']}: {message['content']}" for message in messages])
    
        

class Generate_Prompt:
    """
    Simplified class that uses PromptBuilder for prompt construction,
    mimickiguong the structure of Pairwise_Prompt.
    """
    def __init__(self, model: Any, query: str = None):
        self.query = query or ""
        self.model = model
        #self.promptbuilder = PromptBuilder(model)
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
        # Construct the base prompt (System + User)
        # make_chat_prompt adds generation prompt (e.g. "Assistant:")
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
        self.prompt = message
        """self.prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=True
        )"""
        
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

        You are tasked with determining whether the next step in a mathematical reasoning chain (GEN) **violates or distorts** the logic, assumptions, or conclusions established in the reference claim (REF_STEP). This evaluation will help to identify contradictions, hallucinations, or inconsistencies that may arise between the current reasoning step and the dependency claims it relies on.

        You are given three texts:

        - GLOBAL_PREFIX: The entire set of steps that precede GEN in the reasoning process. This is provided **for context only**, and should only be used to verify contradictions if necessary. It should **not** be used as a constraint when judging the validity of GEN in isolation.
        
        - REF_STEP: A specific prior dependency claim extracted from GLOBAL_PREFIX. This serves as your **local anchor** for comparison. The key task is to determine whether GEN violates, distorts, or contradicts this claim.

        - GEN: The candidate next step, which you must evaluate for consistency against REF_STEP.

        ---

        ## Key Responsibilities

        1. **Your primary task is to evaluate GEN** in relation to REF_STEP. Specifically, you will determine whether GEN introduces contradictions, alters conditions without justification, or misinterprets symbols and conclusions.

        2. **REF_STEP is your normative reference**. All judgments must be made **solely** by comparing GEN to REF_STEP. This is your local contract, and you must assess GEN based on what is explicitly stated or implied in REF_STEP.

        3. **GLOBAL_PREFIX is read-only background**. You may reference GLOBAL_PREFIX **only** to **verify** that any perceived contradiction is not due to a lack of context in REF_STEP and GEN alone. If the perceived conflict can be explained or resolved by GLOBAL_PREFIX, do not treat it as an inconsistency.

        ---

        ## Hard Information Constraints

        You must adhere to the following strict rules:

        1. **Only use the information** explicitly present in:
        - REF_STEP
        - GEN
        - Optionally, you may use GLOBAL_PREFIX to verify whether the perceived conflict between GEN and REF_STEP is due to incomplete information (but **not** to judge directly).

        2. **You must ignore** any of the following:
        - Any steps that occur after REF_STEP, even if they appear in GLOBAL_PREFIX.
        - The original textual problem statement (unless it is explicitly included in REF_STEP).
        - Any background mathematical knowledge that is not explicitly derived from REF_STEP.

        3. **You may use basic logical and algebraic reasoning**, but **only** for the specific expressions, variables, and conditions appearing in REF_STEP and GEN.

        4. If a contradiction is suspected between REF_STEP and GEN, **first check whether this contradiction can be resolved by GLOBAL_PREFIX**. If GLOBAL_PREFIX explains or justifies the conflict, **treat GEN as consistent**.

        5. If you are unsure whether a true contradiction exists, you must **err on the side of consistency** and assign GEN the **higher score**.

        ---

        ## Types of Inconsistencies / Hallucinations (Relative to REF_STEP)

        GEN is considered **inconsistent** or hallucinatory if one or more of the following occurs:

        ### 1. Direct Logical Conflict
        - **GEN reverses, negates, or alters a condition** stated in REF_STEP without any valid explanation or justification.  
            - Example: If REF_STEP states `x > 0`, GEN cannot suddenly assume `x ≤ 0` unless explained or justified.

        ### 2. Symbol or Context Inconsistency
        - A symbol or condition used in both REF_STEP and GEN is given **inconsistent meanings**.  
            - Example: If REF_STEP defines `x` as a positive real number, GEN cannot assume `x` is negative without stating a contradiction or subcase.

        - GEN introduces assumptions that **contradict** the conditions in REF_STEP.  
            - Example: REF_STEP states `x = 0`; GEN then assumes `x ≠ 0` without justifying this shift in meaning.

        ### 3. Ignoring Explicit Constraints
        - REF_STEP gives a crucial condition or assumption (e.g., `n` is even, or `x ≥ 1`), and GEN **proceeds as if this condition does not exist**, making logical conclusions or claims incompatible with it.  
            - Example: REF_STEP states `x ≥ 1`; GEN then proceeds with `x < 1` without considering the constraint.

        ### 4. **Scope Distortion (Weakening or Strengthening Conditions)**
        - GEN **weakens** a condition or conclusion established in REF_STEP. This can lead to an erroneous generalization or misunderstanding of the scope.  
            - Example: If REF_STEP states `x > 0` as a key condition, GEN cannot casually conclude `x ≥ 0` without a proper logical transition or justification.
        
        - GEN **strengthens** a condition without proper justification.  
            - Example: If REF_STEP indicates that `x` could be non-negative (i.e., `x ≥ 0`), GEN assuming `x > 0` without any explanation could mislead the reasoning.

        - **Any distortion of scope** (either weakening or strengthening) must be clearly explained or justified by GEN. If not, it is considered inconsistent.

        ---

        ## What IS Allowed

        GEN **can**:

        - Introduce **subcases or auxiliary reasoning constructs** that do not contradict or weaken any part of REF_STEP.  
            - Example: If REF_STEP establishes a general rule, GEN may introduce a specific case where the rule applies under certain conditions.

        - Introduce **new assumptions or definitions**, provided they do not contradict any explicitly stated or implied assumption in REF_STEP.  
            - Example: If REF_STEP includes a condition `x ≥ 0`, GEN may define a new variable `y = x + 1`, as long as `y ≥ 1` is consistent with `x ≥ 0`.

        - Extend the reasoning in a way that **preserves the validity** of the original conclusions in REF_STEP.  
            - Example: If REF_STEP proves a relationship between `x` and `y`, GEN can explore that relationship further, provided it does not invalidate the original findings.

        ---

        ## Scoring Rules (0–5)

        Your judgment reflects **how severely GEN violates** the consistency and logic of REF_STEP, not whether it is globally incorrect.

        - **5 – Fully Consistent:** GEN fully respects and extends the reasoning of REF_STEP. No contradictions or distortions are introduced. All assumptions and conclusions are consistent with REF_STEP.

        - **4 – Minor Issues:** GEN introduces small ambiguities or unusual inferences, but these do not conflict directly with REF_STEP. There may be slightly radical conclusions, but these are not contradictions.

        - **3 – Weak Consistency:** GEN mostly aligns with REF_STEP but may be poorly argued or introduce some unclear logic. There is no direct contradiction, but parts of GEN are weakly connected.

        - **2 – Significant Conflict:** GEN clearly contradicts or omits important elements from REF_STEP. The reasoning diverges in significant ways but some relevance remains to the original step.

        - **1 – Severe Misalignment:** GEN largely ignores or misuses the conditions or conclusions established in REF_STEP. Most of the reasoning in GEN does not conform to the core findings of REF_STEP.

        - **0 – Direct Contradiction:** GEN fundamentally contradicts REF_STEP (e.g., reversing a proven fact, altering a key equation, or concluding a point that cannot be logically reached based on REF_STEP). The contradiction is irreconcilable and cannot be resolved by GLOBAL_PREFIX.

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
            "- Perform a **pairwise contradiction / hallucination check** between the current step **GEN** and the single prior dependency claim **REF_STEP**.\n"
            "- Treat **REF_STEP as your only normative local context for judging inconsistency**.\n"
            "- You may read GLOBAL_PREFIX only to understand notation, but you MUST NOT use it as extra evidence of conflict.\n"
            "- When uncertain whether there is a conflict based on REF_STEP alone, treat GEN as **consistent** and choose the **higher score**.\n"
            "\n"
            "## REF_STEP (local dependency-claim anchor)\n"
            f"{ref_text}\n\n"
            "## GEN (current step)\n"
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
        #self.promptbuilder = PromptBuilder(model)
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
            "Task: Evaluate GEN against one claim from the matching reference step.\n"
            "Use the reference claim as a local anchor for support, contradiction, and omission checking.\n"
            "Do not use claims from other steps. Do not use outside knowledge.\n"
            "If GEN conflicts with the reference claim, lower the score. If GEN is merely under-justified, score conservatively.\n"
            "Output strictly JSON: {{\"score\": k}} where k ∈ {{0,1,2,3,4,5}}.\n"
            f"REFERENCE_STEP_LABEL:\n{label}\n"
            f"REFERENCE_CLAIM:\n{ref_claim_text}\n"
            f"GEN:\n{gen_text}\n"
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
    
class Judge_Prompt:
    def __init__(self, model: Any):
        self.user_message = ""
        self.system_message = (
            """
            Reasoning: high
            Task Description: You are an automated evaluator. Examine a reference math solution (REF) and a generated solution excerpt (GEN). Determine how faithfully GEN follows REF, focusing on any hallucinations. A hallucination is any content in GEN that is not supported by REF – including factual errors, changes in reasoning structure, or altered context that breaks consistency with REF.
            Hallucination Types to Detect
            Factual Inconsistency: GEN introduces numerical results, formulas, or assumptions that are not logically or explicitly derived from the information in REF.
            Structural Inconsistency: GEN alters the logical flow or method of solution. It changes derivation steps or the solving approach in a way that deviates from REF’s structure.
            Context Inconsistency: GEN omits, adds, or misrepresents critical context from REF (e.g. variables, constraints, or conditions), resulting in a mismatch with the scenario or assumptions in REF.
            Constraints
            Reference (REF): One or more paragraphs of detailed mathematical reasoning (the authoritative solution).
            Generated (GEN): A short solution segment (only a few sentences) purportedly summarizing or partially solving the problem.
            Domain-General: The evaluation must apply to any math domain (algebra, calculus, geometry, etc.), using general mathematical language not tied to a specific field.
            Use Only REF Content: Base all judgments strictly on the content of REF. Do not use outside knowledge or unstated mathematical facts – if GEN relies on any knowledge not present or inferable from REF, treat it as a hallucination.
            Output Format
            JSON Only: Return the final judgment as a JSON object with no extra text. Use the format: {"score": k} where k is an integer from 0 to 5.
            Score Value: A higher score means GEN is more faithful to REF (5 = perfect consistency, 0 = complete hallucination).
            Scoring Rubric
            5 — Fully Consistent: GEN’s statements and logic are entirely supported by REF, with no extraneous steps or assumptions. Every claim in GEN can be directly traced to REF.
            4 — Mostly Consistent: GEN is almost fully faithful to REF. It may have minor rewordings or slight omissions, but these do not alter the meaning or correctness of the solution.
            3 — Generally Aligned: GEN aligns with REF on main points but has gaps or mild extrapolations. Some logical links may be missing or not clearly justified by REF’s content.
            2 — Loose Alignment: GEN is thematically related to REF’s topic but significantly diverges in reasoning or facts. It might introduce an unsupported method or misstate a result from REF.
            1 — Weak Relevance: GEN shows only superficial similarity to REF. It contains clear hallucinations — major steps or claims that are unsupported or contradict REF, omitting critical parts of the solution.
            0 — No Alignment (Complete Hallucination): GEN bears no meaningful correspondence to REF. It fabricates solution steps, uses incorrect facts, or makes assumptions entirely outside the scope of REF.
            Additional Evaluation Rules
            No Unjustified Additions: Do not credit GEN for any extrapolation, generalization, or known formula that isn’t explicitly present or derivable from REF. Extra knowledge, even if correct, counts as hallucination if REF didn’t include it.
            Allow Legitimate Variations: It’s acceptable if GEN uses different notation, reorders steps, or simplifies expressions only when these variations are logically equivalent and clearly inferable from REF’s reasoning.
            Penalize Contradictions: If GEN introduces any claim that conflicts with REF (even a minor contradiction or a changed constraint), reduce the score significantly to reflect the inconsistency.
            Bias to Caution: When in doubt between two adjacent scores, choose the lower (more critical) score. The scoring should err on the side of penalizing potential hallucinations rather than overlooking them.
            Instruction: Read REF and GEN carefully. Identify any hallucinations per the above criteria. Then output the appropriate consistency score (0–5) in the specified JSON format, and nothing else."""
        )
        self.model = model
        self.prompt = ""
        self.promptbuilder = PromptBuilder(model)
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

    def build_user(self, gen_text: str, ref_text: str) -> str:
        self.user_message = (
           "You are given REF (authoritative solution, one or more paragraphs) and GEN (short solution segment, a few sentences)."
            "Judge whether GEN contains hallucinations relative to REF (factual, structural, or context inconsistencies). Use only information present or directly inferable from REF; any outside knowledge counts as hallucination. Evaluate only this segment; ignore any later steps or unstated context."
            "Apply the system rubric. If uncertain between two adjacent levels, choose the lower."
            "Return STRICT JSON ONLY: {'score': <0|1|2|3|4|5>}"
            "No explanations, no extra keys, no code fences."
            f"REF: {ref_text}"
            f"GEN: {gen_text}"
            "Valid outputs: 0,1,2,3,4,5."
        )



    def return_prompt(self) -> str:
        self.prompt = self.promptbuilder.make_chat_prompt(self.system_message, self.user_message)
        return self.prompt
    
    def run(self, gen_claim: str, ref_claim: str) -> dict:
        """Returns a strict JSON: {score: float, label: str, justification: str}"""
        self.build_user(gen_claim, ref_claim)
        prompt = self.return_prompt()
        out = self.model.generate(prompt, self.output_schema)
        
        score = extract_last_score_part(out)
        return score



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
        # self.promptbuilder = PromptBuilder(model)
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


class Important_Claim_Select_Prompt:
    def __init__(self, model: VLLMRunner):
        self.user_message = ""
        self.system_message = (
            "You are a mathematical reasoning analyst. "
            "Task: Given a sequence of claims, select the most important subset. "
            "Important claims are those essential to problem setup, key derivations, "
            "or final conclusions. Remove redundant, decorative, or low-value claims. "
            "Output STRICT JSON only in this format:\n"
            "{\n"
            "  \"important_segments\": [\n"
            "    {\"id\": <int>, \"text\": \"<claim text>\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "Keep original meaning and claim text. Do not paraphrase."
        )
        self.model = model
        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "important_segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "original claim id"
                            },
                            "text": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 160,
                                "description": "selected important claim text"
                            }
                        },
                        "required": ["id", "text"],
                        "additionalProperties": False
                    },
                    "minItems": 0
                }
            },
            "required": ["important_segments"],
            "additionalProperties": False
        }

    def build_user(self, claims: list[dict] | list[str], max_keep: int | None = None) -> None:
        lines = []
        for i, c in enumerate(claims):
            if isinstance(c, dict):
                cid = c.get("id", i)
                txt = str(c.get("text", "")).strip()
            else:
                cid = i
                txt = str(c).strip()
            if txt:
                lines.append(f"[{cid}] {txt}")

        keep_rule = ""
        if isinstance(max_keep, int) and max_keep > 0:
            keep_rule = f"\nSelect at most {max_keep} claims."

        self.user_message = (
            "Select important claims from the list below.\n"
            "Prioritize key assumptions, major transformations, and final conclusions.\n"
            "Avoid redundant or purely explanatory claims."
            f"{keep_rule}\n\n"
            "Claims:\n"
            + "\n".join(lines)
            + "\n\n"
            "Return strict JSON only."
        )

    def return_prompt(self) -> dict:
        sys = self.system_message
        usr = self.user_message + "\nExample: {\"important_segments\": [{\"id\": 1, \"text\": \"...\"}]}"
        return {
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]
        }

    def run(self, claims: list[dict] | list[str], max_keep: int | None = None) -> dict:
        self.build_user(claims, max_keep=max_keep)
        prompt = self.return_prompt()
        out = self.model.generate([prompt], self.output_schema)

        payload = out
        if isinstance(out, tuple) and len(out) >= 2:
            payload = out[1]
        if isinstance(payload, list):
            payload = payload[0] if payload else "{}"
        if isinstance(payload, dict):
            return payload
        return json.loads(payload)


class Progress_Prompt:
    """
    评估 GEN 是否对解题有“实质推进”的打分器（粗粒度版本：0/1/2）。
    输入：PROBLEM（可空）、REF（已有解题前缀）、GEN（当前截取出来的一段前缀）
    输出：{"score": k}，k ∈ {0,1,2}，数值越大代表“推进越明显”。
    """
    def __init__(self, model: VLLMRunner):
        self.model = model
        self.promptbuilder = PromptBuilder(model)
        self.user_message = ""
        self.system_message = (
            """
            Reasoning: high.

            ## Role

            You are an automated evaluator of **REASONING PROGRESS** in step-by-step mathematical solutions.

            Your job is to judge whether a candidate segment **GEN** actually **moves the solution forward**, rather than merely:
            - restating the problem,
            - repeating previously known information,
            - giving vague commentary or reflection without concrete progress.

            You will be given:
            - an optional problem statement **PROBLEM**,
            - a prefix of an existing solution **REF** (all previous steps before GEN),
            - a candidate next segment **GEN** (a truncated prefix from a model's reasoning).

            Your task is to assign a single integer score that reflects **how much GEN contributes to advancing the solution**.

            ---

            ## What Counts as REAL PROGRESS

            GEN shows **genuine progress** if it adds **new, usable mathematical content** that helps to solve the problem, such as:

            1. **New derived relations**
            - Deriving a new equation, inequality, bound, or identity from previous steps.
            - Simplifying an expression in a way that clearly brings it closer to a solvable form.

            2. **Concrete case splits or subgoals**
            - Introducing a meaningful case split (e.g., "case 1: n is even, case 2: n is odd") that is consistent with the problem.
            - Clearly setting up a subproblem or lemma that, if solved, would directly contribute to the main goal.

            3. **Legitimate transformations of the current state**
            - Applying a known theorem or technique to the current expressions in REF.
            - Making a substitution, change of variables, or reparameterization that structures the problem better.

            Progress is **local**: you only need to judge whether GEN moves forward relative to **PROBLEM + REF**, not whether it finishes the entire solution.

            ---

            ## What Does NOT Count as Progress

            GEN should be treated as **little or no progress** when it is mainly:

            1. **Restating the problem or given conditions**
            - Simply repeating the problem statement in different words.
            - Listing assumptions or conditions that are already clearly stated in PROBLEM or REF, without adding anything new.

            2. **Vague reflection or meta-talk**
            - Saying things like “we need to be careful”, “this is a hard problem”, “we should try to use inequality techniques”, without actually applying any technique or deriving something concrete.
            - Generic comments about strategy that do not pin down a specific next move.

            3. **Summarizing instead of advancing**
            - Rephrasing previous steps, summarizing what has been done so far, without introducing any new equations, cases, or subgoals.

            4. **Irrelevant or off-topic content**
            - Talking about unrelated concepts, examples, or comments that do not help solve the given PROBLEM.
            - Chatty or narrative text that does not change the mathematical state of the solution.

            5. **Purely tautological or circular statements**
            - Saying “we want to prove X, so we will try to prove X” without a concrete plan.
            - Restating the goal without bringing in new constraints or structure.

            When GEN is mostly of these forms, you should give a **low score**.

            ---

            ## Scope of Evaluation

            - Use both **PROBLEM** and **REF** as the context for judging progress.
            - You do **not** need to verify full correctness or absence of hallucination.
            - Focus on:
            - whether GEN introduces **new, specific, actionable mathematical content**,
            - whether this content is **plausibly relevant** to solving the problem.

            If GEN contains a mix of fluff and real progress, score based on the **net amount of genuine progress**.

            When you are unsure whether GEN truly advances the solution or just rephrases existing content, you should **choose the lower score** (be conservative in awarding progress).

            ---

            ## Scoring Guide (0–2)

            Give a single integer **"score"** in `{0,1,2}`:

            - **2 – Clear, meaningful progress**
            - GEN makes a concrete, relevant step forward:
                - derives a useful relation,
                - sets up a clear subgoal or case,
                - or performs a non-trivial simplification that helps the solution.
            - A human solver would agree that the solution is **further along** after GEN.

            - **1 – Weak but real progress**
            - GEN adds some **new and relevant** mathematical content, but:
                - the step is small, partially vague, or mixed with a lot of repetition / commentary.
            - There is some forward movement, but it is limited or poorly articulated.

            - **0 – No real progress**
            - GEN is mostly:
                - restatement of the problem or previous steps,
                - vague reflection or meta-talk,
                - summary without new content,
                - or irrelevant / off-topic text.
            - The mathematical state of the solution is essentially **unchanged** after GEN.

            When you are unsure between two scores, choose the **lower score**.

            ---

            ## Output Format (Very Strict)

            Return **only** a single JSON object of the form:

            `{"score": k}`

            where `k` is an integer in `{0,1,2}`.

            Constraints:

            - Do **not** include any explanation, analysis, comments, or natural-language text.
            - Do **not** add keys like `"analysis"` or `"reason"`.
            - Do **not** wrap the JSON in quotes or in code fences in your actual output.
            - The entire reply must be **exactly one JSON object**.
            """
        )
        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [0, 1, 2],
                    "description": "coarse progress score (0–2), higher = stronger genuine progress"
                }
            },
            "required": ["score"],
            "additionalProperties": False
        }

    def build_user(self, gen_text: str, problem_text: str = "", ref_text: str = "") -> None:
        """
        problem_text: 原题，可为空字符串
        ref_text: 当前为止的已知解题前缀（多个step拼起来）
        gen_text: 待测这段GEN（通常是某个前缀截断）
        """
        self.user_message = (
            "## Task\n"
            "Evaluate how much the candidate segment **GEN** makes **genuine progress** in solving the problem,\n"
            "relative to the given **PROBLEM** and the existing solution prefix **REF**.\n"
            "\n"
            "- Reward GEN only for **new, concrete, relevant mathematical content** that moves the solution forward.\n"
            "- Penalize GEN if it is mainly restatement, summary, reflection, or irrelevant text.\n"
            "- When uncertain whether GEN truly advances the solution, choose the **lower score**.\n"
            "\n"
        )

        if problem_text:
            self.user_message += (
                "## PROBLEM (optional)\n"
                f"{problem_text}\n\n"
            )

        if ref_text:
            self.user_message += (
                "## REF (solution prefix so far)\n"
                f"{ref_text}\n\n"
            )

        self.user_message += (
            "## GEN (candidate segment to evaluate)\n"
            f"{gen_text}\n\n"
            "## Output Requirements\n"
            "- Output a single strict JSON object: `{\"score\": k}` where `k ∈ {0,1,2}`.\n"
            "- Valid outputs: `0, 1, 2`.\n"
        )

    def return_prompt(self) -> str:
        self.prompt = self.promptbuilder.make_chat_prompt(self.system_message, self.user_message)
        return self.prompt

    def run(self, gen: str, problem: str = "", ref: str = "") -> dict:
        """
        评估一段 GEN 的“推进程度”（粗粒度：0/1/2）。
        problem: 原题文本（可空）
        ref: 现有解题前缀（可空）
        gen: 待测片段
        """
        self.build_user(gen_text=gen, problem_text=problem, ref_text=ref)
        prompt = self.return_prompt()
        out = self.model.generate(prompt, self.output_schema)
        score = extract_last_score_part(out[0])
        return {
            "score": score,
            "raw_output": out,
            "gen": gen,
            "problem": problem,
            "ref": ref,
        }


class Claim_Dependency_Prompt:
    """
    仅判断方向 A -> B：Claim A 是否依赖 Claim B。
    无评分，仅返回结论与简短解释。
    """
    def __init__(self, model: VLLMRunner):
        self.model = model
        self.user_message = ""
        self.system_message = (
            """
            You are an expert at identifying direct proof dependencies between claims in a mathematical or logical argument.

Task:
Determine whether Claim A directly depends on Claim B.

Target notion:
A directly depends on B iff B is one of the minimal, local parent claims needed to interpret or justify A in the proof.

Important:
You are NOT judging whether B is vaguely relevant, topically related, or part of the same overall proof.
You are judging whether B is a direct parent of A in the local dependency graph.

Return "yes" only if at least one of the following holds:

1. Direct inferential parent
   B is an immediately used premise, lemma, or intermediate result from which A is directly concluded.

2. Essential object/construction introduction
   A uses an object, symbol, line, segment, circle, ratio, or construction whose mathematical role is first established in B,
   and without B, A would be incomplete, referentially unclear, or not well-formed in the local proof context.

3. Essential semantic prerequisite
   Even if A does not literally repeat B's wording, B establishes the exact mathematical entity or property that A directly relies on.
   Examples:
   - "H is the incenter of triangle DEF" can directly support claims about the incircle of triangle DEF.
   - "A is the D-excenter of triangle DEF" can directly support claims about the D-excircle of triangle DEF.
   Lexical mismatch does NOT rule out dependency if the semantic role is direct and necessary.

4. Direct definition / criterion / theorem trigger
   B provides the precise definition, criterion, or theorem-instantiating condition that is directly invoked in A.

Return "no" if any of the following holds:

1. B is only broad background, setup, motivation, roadmap, or a goal statement.
2. B is only globally true in the proof but not directly used for A.
3. B is only a distant ancestor, while a closer prior claim more directly supports A.
4. B shares objects, symbols, or topic with A, but is not actually needed to derive or interpret A.
5. B gives extra positional/detail information about an object already usable in A, but does not establish the role A needs.
6. B is only explanatory narrative, intuition, commentary, or redundant detail.
7. A can still be locally justified or understood without B once the more direct prior claims are available.

Priority rule: prefer the closest sufficient parent
If a later prior claim already directly supports A, do NOT mark an earlier background or ancestor claim as dependency unless A still directly needs both.

Object-introduction rule:
Do not mark every earlier mention of an object as dependency.
Mark "yes" only when B is the claim that gives that object the specific role needed by A.
Example:
- If A uses "segment EF", then B must do more than merely mention E or F somewhere;
  B must help establish the relevant construction or role of E/F needed for A.

Semantic-link rule:
A may depend on B even without shared surface words, if B directly establishes the mathematical notion invoked by A.
But do NOT use broad world knowledge to invent missing links; the link must be specific and local to the proof context.

Decision procedure:
Step 1. Ask: what are the most local parent claims actually needed to justify or interpret A?
Step 2. Check whether B is one of those minimal local parents.
Step 3. Reject B if it is only background, a distant ancestor, or extra detail.
Step 4. If a closer prior claim already subsumes B's role for A, prefer "no".
Step 5. Use "uncertain" only when the wording truly does not allow a reliable decision.

Output strictly in JSON format:
{
  "conclusion": "yes" or "no" or "uncertain",
  "explanation": "Briefly state whether B is a minimal direct parent of A, an essential object/semantic prerequisite, or not a direct dependency."
}
            """
        )
        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "conclusion": {
                    "type": "string",
                    "enum": ["yes", "no", "uncertain"],
                    "description": "whether Claim A depends on Claim B under directional judgment A->B"
                },
                "explanation": {
                    "type": "string",
                    "minLength": 1,
                    "description": "brief rationale indicating whether B is necessary premise/condition/definition/support for A"
                }
            },
            "required": ["conclusion", "explanation"],
            "additionalProperties": False
        }

    def build_user(self, claim_a: str, claim_b: str) -> None:
        self.user_message = (
            "Input:\n"
            "Claim A: "
            f"{claim_a}\n\n"
            "Claim B: "
            f"{claim_b}\n\n"
            "Only assess direction A -> B.\n"
            "Return strict JSON only with keys: conclusion, explanation."
        )

    def return_prompt(self) -> str:
        sys = self.system_message
        usr = self.user_message
        return {
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]
        }

    def run(self, claim_a: str, claim_b: str) -> dict:
        self.build_user(claim_a, claim_b)
        prompt = self.return_prompt()
        out = self.model.generate(prompt, self.output_schema)

        parsed = safe_json_loads(out[0]) if isinstance(out, list) and out else safe_json_loads(out)
        if isinstance(parsed, dict):
            return {
                "conclusion": str(parsed.get("conclusion", "uncertain")),
                "explanation": str(parsed.get("explanation", "")),
                "raw_output": out,
                "claim_a": claim_a,
                "claim_b": claim_b,
            }

        return {
            "conclusion": "uncertain",
            "explanation": "Unable to parse model output into required JSON schema.",
            "raw_output": out,
            "claim_a": claim_a,
            "claim_b": claim_b,
        }
