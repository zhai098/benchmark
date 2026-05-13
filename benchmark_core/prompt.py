from __future__ import annotations

from benchmark_core.config import Config
from typing import Any
from benchmark_core.data_process import safe_json_loads, extract_last_score_part, extract_prefix  # 文件顶部集中导入一次


class GeneratePromptFormatter:
    """
    Generation prompts are vLLM-only in this workflow.
    The formatter deliberately does not render tokenizer templates; VLLMRunner
    owns model-specific assistant-continuation rendering in one place.
    """

    def __init__(self, model: Any):
        self.model_name = getattr(model, "model_name", "")

    def render(self, messages: list[dict[str, Any]]) -> Any:
        return [dict(message) for message in messages]

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
        self.system_message = (
        """
        ## Role

        You are a strict evaluator for local dependency faithfulness in a
        mathematical proof.

        ## **Key Information**

        > **Primary target:** judge whether CURRENT_STEP faithfully preserves,
        > uses, or locally continues DEPENDENCY_CLAIM.

        > **Hallucination focus:** this route should detect dependency-level
        > hallucination, distortion, contradiction, condition loss, or unsupported
        > strengthening of the dependency claim.

        > **Non-use is not automatically hallucination.** If CURRENT_STEP does
        > not clearly use DEPENDENCY_CLAIM but also does not contradict, distort,
        > or misuse it, treat this as weak dependency evidence and usually score 3.

        > **Additional context is allowed.** CURRENT_STEP may use other prior
        > facts or continue the proof beyond DEPENDENCY_CLAIM, provided it does
        > not change, misuse, or conflict with DEPENDENCY_CLAIM.

        > **Do not reward mere compatibility.** If CURRENT_STEP is only weakly
        > related to DEPENDENCY_CLAIM, phase-misaligned, or jumps beyond local
        > support, score at most 3.

        > **Hard failure:** if CURRENT_STEP changes, misuses, or contradicts
        > DEPENDENCY_CLAIM, lower the score according to severity.

        You are given three texts:

        - GLOBAL_PREFIX: earlier reference steps before CURRENT_STEP. Use it
        only to understand notation or disambiguate symbols. Do not use it as
        an additional source of proof obligations.

        - DEPENDENCY_CLAIM: the single local claim that CURRENT_STEP is
        supposed to depend on.

        - CURRENT_STEP: the generated reasoning step to evaluate.

        Your task is not to judge whether CURRENT_STEP solves the whole
        problem. Your task is to judge whether CURRENT_STEP faithfully
        preserves, uses, or locally continues DEPENDENCY_CLAIM.

        ## Decision Priorities

        Judge in this order:

        1. Does CURRENT_STEP contradict, reverse, weaken, strengthen, or change
        the meaning of DEPENDENCY_CLAIM?
        2. Does CURRENT_STEP preserve the symbols, conditions, quantifiers, and
        scope of DEPENDENCY_CLAIM?
        3. Does CURRENT_STEP actually use or locally continue DEPENDENCY_CLAIM?
        4. Does CURRENT_STEP make a conclusion that DEPENDENCY_CLAIM does not
        locally support?
        5. Is CURRENT_STEP merely compatible with DEPENDENCY_CLAIM but not
        meaningfully connected to it?

        ## Important Scoring Constraints

        - Mere compatibility is not enough for score 5.
        - If CURRENT_STEP is compatible with DEPENDENCY_CLAIM but the
        dependency relation is weak, unclear, or only implicit, score at most 3.
        - If CURRENT_STEP does not clearly use DEPENDENCY_CLAIM but also does
        not contradict, distort, or misuse it, this is weak dependency evidence;
        score 3 rather than treating it as a severe hallucination.
        - If CURRENT_STEP relies on other prior facts, do not penalize this by
        itself. Penalize only if it changes, misuses, ignores in a harmful way,
        or conflicts with DEPENDENCY_CLAIM.
        - If CURRENT_STEP jumps directly to a broad or final conclusion that is
        not locally supported by DEPENDENCY_CLAIM, score at most 3.
        - If CURRENT_STEP uses DEPENDENCY_CLAIM but omits an important
        condition or changes its scope, score at most 2.
        - If CURRENT_STEP directly contradicts DEPENDENCY_CLAIM, score 0 or 1.
        - Use GLOBAL_PREFIX only to clarify notation. Do not use it to rescue
        a distorted, contradicted, or unsupported use of DEPENDENCY_CLAIM.
        - When uncertain between adjacent scores, choose the lower score unless
        CURRENT_STEP clearly and faithfully uses DEPENDENCY_CLAIM.

        ## What Counts as Faithful Use

        CURRENT_STEP faithfully uses DEPENDENCY_CLAIM when it:

        - preserves the claim's exact conditions and scope;
        - applies the claim to the same objects or symbols, or explicitly maps
        them without distortion;
        - makes a local inference that follows from the claim;
        - does not turn a local claim into a stronger global conclusion without
        support.

        ## Scoring Rules

        Give an integer score from 0 to 5.

        5: Strong faithful continuation. CURRENT_STEP clearly uses
        DEPENDENCY_CLAIM, preserves its meaning and scope, and makes a valid
        local continuation.

        4: Mostly faithful. CURRENT_STEP uses or preserves DEPENDENCY_CLAIM
        with a minor omission, mild compression, or small under-justified
        transition, but no meaningful distortion.

        3: Weak dependency evidence. CURRENT_STEP is compatible with
        DEPENDENCY_CLAIM but only weakly related, does not clearly use it,
        moves beyond it using other context, is under-justified, or is
        phase-misaligned, without contradiction or distortion.

        2: Significant problem. CURRENT_STEP partially misuses
        DEPENDENCY_CLAIM, drops an important condition, makes an unsupported
        local leap, or changes scope in a way that affects the reasoning.

        1: Severe misuse. CURRENT_STEP largely distorts or misapplies
        DEPENDENCY_CLAIM, or makes claims that depend on it while ignoring an
        essential condition.

        0: Direct contradiction. CURRENT_STEP fundamentally contradicts
        DEPENDENCY_CLAIM.

        ## Output Format

        Return only one JSON object:

        `{"score": k}`

        where k is one of 0, 1, 2, 3, 4, 5.

        Do not output explanations, comments, markdown, or additional keys.

        """)


        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [0, 1, 2, 3, 4, 5],
                    "description": "discrete consistency score (0-5), higher = more consistent"
                }
            },
            "required": ["score"],
            "additionalProperties": False
        }
        self.global_prefix = ""

    def set_global_prefix(self, prefix: str | None):
        self.global_prefix = prefix or ""

    def build_user(self, gen_text: str, ref_text: str, prefix: str | None = None) -> None:
        if prefix is None:
            prefix = self.global_prefix

        self.user_message = (
            "## GLOBAL_PREFIX (background only; DO NOT use it as a source of constraints)\n"
            f"{prefix}\n\n"
            "## Task\n"
            "- Evaluate whether CURRENT_STEP faithfully preserves, uses, or locally continues the single DEPENDENCY_CLAIM.\n"
            "- DEPENDENCY_CLAIM is the only normative local anchor for this pairwise check.\n"
            "- GLOBAL_PREFIX may be used only to understand notation or resolve local ambiguity.\n"
            "- Do not reward mere compatibility as score 5. If the relation to DEPENDENCY_CLAIM is weak, unclear, or phase-misaligned, score at most 3.\n"
            "- If CURRENT_STEP does not clearly use DEPENDENCY_CLAIM but does not contradict, distort, or misuse it, treat this as weak dependency evidence and score 3.\n"
            "- CURRENT_STEP may rely on other prior facts; penalize only if it changes, misuses, or conflicts with DEPENDENCY_CLAIM.\n"
            "- If CURRENT_STEP jumps to a broad/final conclusion not locally supported by DEPENDENCY_CLAIM, score at most 3.\n"
            "- If CURRENT_STEP changes, weakens, strengthens, or contradicts DEPENDENCY_CLAIM, lower the score according to severity.\n"
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
        prompts = []
        scores = []

        for ref_step in ref:
            self.build_user(gen_claim, ref_step, prefix=prefix)
            prompt = self.return_prompt()
            prompts.append(prompt)

        reasonings, outs = self.model.generate(prompts, None)

        for out in outs:
            score = extract_last_score_part(out)
            scores.append(score)

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
            ## Role

            You are tasked with evaluating the **continuity of reasoning structure** in a step-by-step mathematical solution.

            ## **Key Information**

            > **Primary target:** judge whether GEN faithfully continues REF's method, proof phase, and active subgoal.

            > **This is a structural route.** Do not require a full correctness proof, but do not reward a structurally similar step that is not faithful to REF.

            > **Important caps:** broad/final conclusion jumps, changed targets/phases, or explicit contradictions to REF commitments should be capped according to the guardrails below.

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

            ### 5. Light Fidelity Guardrails

            This route primarily evaluates structural continuation, not full mathematical correctness. However, structural continuity must be faithful to REF.

            - Do not give score 5 merely because GEN uses similar vocabulary or the same general topic.

            - GEN must continue the active proof phase and subgoal established by REF.

            - Do not penalize a small amount of repetition or local restatement when GEN stays faithful to REF and continues in the correct direction.

            - If GEN jumps directly to a broad or final conclusion without continuing the missing intermediate reasoning, score at most 2.

            - If GEN changes the target theorem, bound, construction, case, or proof phase established by REF, score at most 2.

            - If GEN contradicts an explicit commitment in REF, such as a symbol meaning, assumption, case condition, or active subgoal, score at most 2.

            - Minor local arithmetic or algebra slips that do not affect the reasoning flow may still receive 4, but not 5 if they obscure the next-step connection.

            ---

            ## Scoring (0–5)

            Give an integer score `"score"`. `{0,1,2,3,4,5}`:

            - **5 – Excellent Structural Continuity**  
            GEN follows REF’s plan, active proof phase, and current subgoal. The logic is well-connected, GEN makes **significant next-step progress**, and it does not break explicit commitments from REF.

            - **4 – Good Structural Continuity, but with Minor Issues**  
            GEN largely follows the reasoning in REF and is logically coherent, with **minor issues** (e.g., slight vagueness, a minor local slip, or a slightly rushed transition). The overall structure and proof phase remain intact.

            - **3 – Weak Structural Continuity, but Acceptable**  
            GEN maintains a connection to REF, but the reasoning is vague, stalled, mostly repetitive without new progress, or only loosely connected. GEN adds limited value in terms of **advancing the solution**.

            - **2 – Questionable Structure**  
            GEN diverges noticeably from the reasoning in REF. The logical connections are **weak**, and GEN introduces issues like **route/target/phase shifts**, unsupported final jumps, irrelevant steps, or broken REF commitments.

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
            "Judge whether **GEN** faithfully continues the same method, proof phase, and active subgoal committed in **REF** (all prior steps).\n"
            "Penalize **route switching**, unproductive repetition without forward progress, jumping ahead, changing the target, or breaking explicit REF commitments.\n"
            "Focus on structural continuation. Do not require a full correctness proof, but do not reward a structurally similar step that is not faithful to REF.\n"
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
        self.system_message_without_reference = (
            """
            ## Role:
            You are an automated evaluator for reference-free mathematical hallucination risk,
            internal soundness, and self-consistency.

            ## **Key Information**

            > **Primary target:** judge whether GEN contains detectable mathematical
            > hallucination, self-contradiction, illegal inference, or internally unsupported
            > assertions presented as established facts.

            > **Do not over-penalize incompleteness.** A missing derivation is not by itself
            > hallucination unless GEN asserts an unsupported fact as established, relies on a
            > false or unstated assumption, contradicts itself, or makes an invalid inference.

            > **Do not give full credit too easily.** If the direction is plausible but a key
            > derivation is missing, the score should usually be 3. If only routine details are
            > omitted, the score can be 4. Score 5 requires strong internal support.

            ## Task

            Inspect GEN for internal mathematical correctness and self-consistency. Identify
            arithmetic/algebraic mistakes, illegal operations, undefined or redefined symbols,
            incompatible constraints, unsupported final conclusions, and self-contradictions.

            ## Rules:
            - Use only GEN. Do not use claims from other steps or the original problem statement.

            - You may use basic mathematical facts and direct consequences of definitions
            explicitly present in GEN. For example, you may check algebra, counting, domain
            restrictions, quantifiers, and standard implications of stated definitions.
            
            - Focus on immediate internal coherence:
              - **Variable definitions**
              - **Domain restrictions**
              - **Equation manipulations**
              - **Sign/inequality directions**
              - **Step-to-step consistency within GEN**
              - **Counting or set-size claims**
              - **Whether central claims are supported by preceding lines**
            
            - If GEN jumps to a final answer, broad theorem, or major conclusion without the
            needed derivation inside GEN, but there is no detectable contradiction, false step,
            or invalid assumption, score at most 3.

            - If GEN's unsupported conclusion relies on a false step, contradicts earlier text,
            changes conditions, or introduces an unjustified assumption as fact, score at most 2.

            - If a central mathematical, algebraic, counting, or constraint error appears in GEN, score at most 2.

            - If a result is plausible but a key derivation is missing or unclear, score at most 3.

            - If GEN omits only routine intermediate details while the core reasoning is clear
            and internally sound, score at most 4.

            - Do not penalize a long answer or small amount of repetition by itself. If the
            direction is correct and the reasoning is internally sound, only reduce the score
            for real loss of focus, missing support, unsupported assertions, or errors.

            - When uncertain between two adjacent scores, choose the lower score unless the
            evaluated text clearly justifies the higher one.

            ## Output (strict):
            - **JSON only**: `{"score": k}` where `k` ∈ `{0,1,2,3,4,5}`; higher = lower
            detectable hallucination risk and stronger internal soundness.

            ## Scoring Guide:
            - **5**: Core reasoning is internally sound, symbols and constraints are consistent,
            and central conclusions are sufficiently derived. No detectable hallucination.
            - **4**: Mostly sound, with only routine omitted details, mild under-justification,
            or small expression issues that do not affect the central reasoning.
            - **3**: Plausible direction and no clear hallucination, but a key derivation is
            missing/unclear, or a central conclusion is under-supported.
            - **2**: Central mathematical error, unsupported conclusion relying on a false or
            unstated assumption, conflicting constraint, or invalid inference.
            - **1**: Multiple central errors; reasoning is largely unsound or hallucinated.
            - **0**: Nonsensical or self-contradictory throughout.

            ## Instruction:
            Output only `{"score": k}`.
        """)
        self.system_message_with_reference = (
            """
            ## Role:
            You are an automated evaluator for same-step claim faithfulness, mathematical
            hallucination risk, and internal soundness.

            ## **Key Information**

            > **Primary target:** first judge whether CURRENT_STEP covers, preserves, or
            > correctly uses CURRENT_STEP_CLAIM.

            > **Neutral non-coverage:** if CURRENT_STEP does not cover CURRENT_STEP_CLAIM
            > but also does not contradict, distort, or misuse it, fall back to internal
            > soundness and score at most 3.

            > **Do not over-penalize incompleteness.** If CURRENT_STEP is directionally
            > correct but the proof is incomplete, do not give a low score solely for
            > incompleteness. However, do not give full credit unless key support is present.

            > **Hallucination focus:** penalize false assertions, changed conditions,
            > symbol/scope distortion, unsupported assumptions presented as facts, and
            > contradictions.

            ## Task

            Evaluate CURRENT_STEP against one CURRENT_STEP_CLAIM from the same generated
            step. The claim is a local anchor for support, contradiction, omission, and misuse
            checking.

            ## Rules:
            - Use only CURRENT_STEP and CURRENT_STEP_CLAIM. Do not use claims from other
            steps or the original problem statement.

            - First check whether CURRENT_STEP covers, preserves, or correctly uses
            CURRENT_STEP_CLAIM.

            - If CURRENT_STEP mentions claim-related content but misquotes it, changes symbols,
            drops conditions, changes scope, or uses the claim incorrectly, score at most 2.

            - If CURRENT_STEP does not cover CURRENT_STEP_CLAIM and does not conflict with it,
            treat this as neutral non-coverage. Fall back to judging CURRENT_STEP's internal
            correctness, but do not give a score above 3.

            - If CURRENT_STEP is in the correct direction but incomplete or compressed, do not
            score below 3 unless there is a false step, contradiction, condition change, or
            unsupported new assumption.

            - If CURRENT_STEP covers CURRENT_STEP_CLAIM but a key derivation for the central
            conclusion is missing, score at most 3.

            - If CURRENT_STEP covers CURRENT_STEP_CLAIM and omits only routine intermediate
            details while remaining faithful and internally sound, score at most 4.

            - A score of 5 requires CURRENT_STEP to cover and correctly use CURRENT_STEP_CLAIM,
            remain internally sound, and sufficiently derive its central conclusion.

            - You may use basic mathematical facts and direct consequences of definitions
            explicitly present in CURRENT_STEP. For example, you may check algebra, counting,
            domain restrictions, quantifiers, and standard implications of stated definitions.

            - If CURRENT_STEP jumps to a final answer, broad theorem, or major conclusion
            without the needed derivation inside CURRENT_STEP, but there is no detectable
            contradiction, false step, or invalid assumption, score at most 3.

            - If the unsupported conclusion relies on a false step, contradicts earlier text,
            changes conditions, or introduces an unjustified assumption as fact, score at most 2.

            - If a central mathematical, algebraic, counting, or constraint error appears in
            CURRENT_STEP, score at most 2.

            - Do not penalize a long answer or small amount of repetition by itself. If the
            direction is correct and the reasoning is internally sound, only reduce the score
            for real loss of focus, missing support, unsupported assertions, or errors.

            - When uncertain between two adjacent scores, choose the lower score unless
            CURRENT_STEP clearly justifies the higher one.

            ## Output (strict):
            - **JSON only**: `{"score": k}` where `k` ∈ `{0,1,2,3,4,5}`; higher = better
            same-step claim faithfulness and lower hallucination risk.

            ## Scoring Guide:
            - **5**: CURRENT_STEP covers and correctly uses CURRENT_STEP_CLAIM, remains
            internally sound, and sufficiently derives its central conclusion.
            - **4**: CURRENT_STEP covers CURRENT_STEP_CLAIM and is mostly sound, with only
            routine omitted details, mild under-justification, or small expression issues.
            - **3**: CURRENT_STEP does not cover CURRENT_STEP_CLAIM but is internally plausible;
            or it covers the claim only weakly; or the direction is plausible but a key derivation
            is missing.
            - **2**: Clear misuse of CURRENT_STEP_CLAIM, central mathematical error, unsupported
            conclusion relying on a false or unstated assumption, condition/scope change, or
            conflicting constraint.
            - **1**: Severe distortion of CURRENT_STEP_CLAIM or multiple central errors.
            - **0**: Direct contradiction, nonsensical reasoning, or self-contradiction throughout.

            ## Instruction:
            Output only `{"score": k}`.
        """)
        self.system_message = self.system_message_without_reference

        self.prompt = ""
        self.output_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [0, 1, 2, 3, 4, 5],
                    "description": "discrete degree of entailment (0-5), higher = stronger entailment"
                }
            },
            "required": [
                "score"
            ],
            "additionalProperties": False
        }
        
    def build_user_without_reference(self, gen_text: str) -> None:
        self.system_message = self.system_message_without_reference
        self.user_message = (
            "Task: Reference-free evaluation of GEN's internal mathematical soundness and hallucination risk.\n"
            "Check symbol definitions, domain/constraint compatibility, legality of operations, sign/inequality directions, "
            "counting/set-size claims, and step-to-step coherence within GEN.\n"
            "Use only GEN. You may use basic mathematical facts and direct consequences of definitions explicitly stated in GEN.\n"
            "Do not treat missing derivation as hallucination by itself. It becomes serious only when GEN asserts an unsupported fact as established, "
            "relies on a false or unstated assumption, contradicts itself, or makes an invalid inference.\n"
            "If GEN contains a central mathematical, algebraic, counting, or constraint error, score at most 2.\n"
            "If GEN jumps to a final answer, broad theorem, or major conclusion without needed derivation but has no clear false step or contradiction, score at most 3.\n"
            "If GEN is plausible but a key derivation is missing or unclear, score at most 3.\n"
            "If only routine details are omitted and the core reasoning is internally sound, score at most 4.\n"
            "Do not penalize length or a small amount of repetition by itself; penalize only loss of focus, missing support, unsupported assertions, or errors.\n"
            "A score of 5 requires core reasoning to be sound and sufficiently derived, with no detectable hallucination.\n"
            "When uncertain between two scores, choose the lower.\n"
            "Output strictly JSON: {{\"score\": k}} where k ∈ {{0,1,2,3,4,5}}.\n"
            "GEN:\n"
            f"{gen_text}\n"
            "Valid outputs: 0,1,2,3,4,5."
        )

    def build_user_with_reference(self, gen_text: str, ref_claim_text: str, step_label: str | None = None) -> None:
        label = step_label or "same reference step"
        self.system_message = self.system_message_with_reference
        self.user_message = (
            "Task: Evaluate CURRENT_STEP against one claim from the same current step.\n"
            "First check whether CURRENT_STEP covers, preserves, or correctly uses CURRENT_STEP_CLAIM.\n"
            "If CURRENT_STEP mentions claim-related content but misquotes it, changes symbols, drops conditions, changes scope, or uses it incorrectly, score at most 2.\n"
            "If CURRENT_STEP does not cover CURRENT_STEP_CLAIM and does not conflict with it, treat this as neutral non-coverage; fall back to internal correctness, but score at most 3.\n"
            "If CURRENT_STEP is directionally correct but incomplete or compressed, do not score below 3 unless there is a false step, contradiction, condition change, or unsupported new assumption.\n"
            "If CURRENT_STEP covers CURRENT_STEP_CLAIM but misses a key derivation for its central conclusion, score at most 3.\n"
            "If CURRENT_STEP covers CURRENT_STEP_CLAIM and omits only routine intermediate details while remaining faithful and internally sound, score at most 4.\n"
            "A score of 5 requires CURRENT_STEP to cover and correctly use CURRENT_STEP_CLAIM, remain internally sound, and sufficiently derive its central conclusion.\n"
            "If CURRENT_STEP jumps to a final answer, broad theorem, or major conclusion without needed derivation but has no clear false step or contradiction, score at most 3.\n"
            "If CURRENT_STEP contains a central mathematical, algebraic, counting, or constraint error, score at most 2.\n"
            "Do not use claims from other steps or the original problem statement. You may use basic mathematical facts and direct consequences of definitions explicitly stated in CURRENT_STEP.\n"
            "When uncertain between two scores, choose the lower.\n"
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
        "You are a mathematical expert in natural language understanding. "
        "Task: Given a sentence or paragraph from a mathematical solution, "
        "segment it into claim-level propositions.\n"
        "Use a practical granularity: each proposition should be a clear, "
        "independent mathematical claim, but do not over-segment connected "
        "reasoning into tiny fragments. Keep definitions, formulas, quantified "
        "conditions, and short explanatory clauses together when separating them "
        "would make the claim harder to understand.\n"
        "Follow these rules:\n"
        "1. Each proposition must be a clear, independent statement.\n"
        "2. Split only when the text contains multiple logically independent claims.\n"
        "3. Do not paraphrase; preserve the original meaning and notation as much as possible.\n"
        "4. Keep the propositions in order of appearance and number ids from 0.\n"
        "5. Do not impose or mention any fixed number of propositions.\n"
        "6. Output STRICT JSON only, with the following format:\n"
        "{"
        "\"segments\": [\n"
        "  {\"id\": <int>, \"text\": \"<claim-level proposition>\"},\n"
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
                        "description": "claim-level proposition; preserve original meaning and notation"
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
            f"Segment the following mathematical solution into claim-level propositions:\n{text}\n"
        )
    def return_prompt(self) -> str:
        sys = self.system_message
        usr = self.user_message + "\n\nRemember: output JSON only. Example: {\"segments\": [{\"id\": 0, \"text\": \"Claim 1.\"}, {\"id\": 1, \"text\": \"Claim 2.\"}]}"
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
