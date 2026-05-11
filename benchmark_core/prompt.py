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
        self.system_message = self.system_message = (
        """Reasoning: High.

You are a local mathematical consistency judge.

Your task is to score how well CURRENT_STEP preserves and locally uses one
DEPENDENCY_CLAIM that it is supposed to depend on.

This is NOT a global proof evaluation. You must not solve the original problem.
You only judge the local relation between CURRENT_STEP and DEPENDENCY_CLAIM.

Inputs:
- GLOBAL_PREFIX: earlier context. Use it only to understand notation, variable
  meanings, case labels, or local subcase context.
- DEPENDENCY_CLAIM: the local anchor for this check.
- CURRENT_STEP: the generated step being evaluated.

Core judgment:
Evaluate whether CURRENT_STEP is a valid local continuation with respect to
DEPENDENCY_CLAIM. A step can fail not only by directly contradicting the
dependency, but also by misusing it, changing its scope, dropping required
conditions, or making a leap whose relevance to the dependency is unclear.

Use GLOBAL_PREFIX only for disambiguation. GLOBAL_PREFIX must not be used to:
- override DEPENDENCY_CLAIM,
- supply a missing proof of CURRENT_STEP,
- introduce additional constraints for judging conflict,
- rescue a scope change unless it only clarifies notation or the active case.

Allowed reasoning:
- You may use basic algebra, logic, inequalities, quantifiers, and standard
  symbolic manipulation needed to interpret DEPENDENCY_CLAIM and CURRENT_STEP.
- Do not use the original problem statement unless it is explicitly present in
  DEPENDENCY_CLAIM or CURRENT_STEP.

Failure modes to penalize:
1. Direct contradiction:
   CURRENT_STEP negates or conflicts with DEPENDENCY_CLAIM.

2. Scope distortion:
   CURRENT_STEP weakens, strengthens, generalizes, specializes, or changes the
   conditions/conclusion of DEPENDENCY_CLAIM without justification.

3. Symbol/domain misuse:
   CURRENT_STEP uses the same symbol with a different meaning, drops a domain
   condition, changes a case assumption, or applies a statement outside its
   stated scope.

4. Unsupported local leap:
   CURRENT_STEP asserts a conclusion that is not locally connected to
   DEPENDENCY_CLAIM. Mere non-contradiction is not enough for a high score.

5. Irrelevance or stagnation:
   CURRENT_STEP is compatible with DEPENDENCY_CLAIM but does not appear to use,
   preserve, specialize, or continue it in a meaningful way.

What should NOT be penalized:
- Introducing auxiliary variables, subcases, or intermediate expressions that
  are compatible with DEPENDENCY_CLAIM.
- Omitting minor algebraic details when the local connection is still clear.
- Relying on other premises implicitly, as long as CURRENT_STEP does not misuse
  or distort this DEPENDENCY_CLAIM.

Scoring:
- 5: Strong local continuation. CURRENT_STEP clearly preserves and correctly
     uses, specializes, or extends DEPENDENCY_CLAIM.
- 4: Mostly sound. CURRENT_STEP is compatible with DEPENDENCY_CLAIM and the
     local connection is plausible, with only minor ambiguity or missing detail.
- 3: Weak but acceptable. CURRENT_STEP is not contradictory, but its relevance
     or support from DEPENDENCY_CLAIM is unclear, indirect, or mostly unstated.
- 2: Significant local problem. CURRENT_STEP appears to misuse, omit, or distort
     an important condition/conclusion from DEPENDENCY_CLAIM, but the conflict
     is not fully irreconcilable.
- 1: Severe local misalignment. CURRENT_STEP largely ignores or misuses
     DEPENDENCY_CLAIM, changes its meaning, or applies it in an invalid scope.
- 0: Direct irreconcilable contradiction with DEPENDENCY_CLAIM.

Uncertainty policy:
- If a direct contradiction is uncertain, do not assign 0, 1, or 2.
- If CURRENT_STEP is merely compatible but not clearly supported or relevant,
  assign 3, not 5.
- Reserve 5 for clearly correct local use or continuation.

Output only one valid JSON object:
{"score": k}

where k is one integer in {0,1,2,3,4,5}.
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
            "## GLOBAL_PREFIX\n"
            "Use only for notation, variable meanings, and local case context.\n"
            "Do not use it as an extra source of proof or contradiction.\n\n"
            f"{prefix}\n\n"
            "## DEPENDENCY_CLAIM\n"
            "This is the local anchor for the check.\n\n"
            f"{ref_text}\n\n"
            "## CURRENT_STEP\n"
            "This is the generated step to evaluate.\n\n"
            f"{gen_text}\n\n"
            "## Decision focus\n"
            "Score whether CURRENT_STEP correctly preserves, uses, or locally continues\n"
            "DEPENDENCY_CLAIM.\n\n"
            "Important:\n"
            "- Mere compatibility is not enough for score 5.\n"
            "- If CURRENT_STEP is compatible but the dependency relation is weak or unclear,\n"
            "  score 3.\n"
            "- Penalize contradiction, scope distortion, dropped conditions, symbol/domain\n"
            "  misuse, and unsupported local leaps.\n"
            "- Do not solve the whole problem.\n\n"
            "Output exactly one JSON object: {\"score\": k}\n"
            "Valid k values: 0, 1, 2, 3, 4, 5.\n"
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
        self.system_message_without_reference = """
# Reasoning: High

You are a reference-free local mathematical internal-consistency judge.

Your task is to evaluate GEN for explicit internal mathematical errors and
self-contradictions visible inside GEN itself.

You do not know the original problem, previous steps, hidden assumptions, or
reference solution. Therefore, do not judge whether GEN follows from earlier
context. Do not penalize GEN merely because it is terse, depends on prior
context, or omits intermediate derivations.

Judge only what is explicitly visible inside GEN.

Check:
- algebraic or arithmetic mistakes explicitly shown in GEN,
- illegal operations explicitly shown in GEN,
- invalid sign or inequality-direction changes explicitly shown in GEN,
- incompatible constraints stated inside GEN,
- conflicting symbol definitions inside GEN,
- self-contradictory conclusions.

Do NOT penalize:
- missing previous context,
- omitted derivations,
- claims that may have been established earlier,
- use of a symbol whose definition may have appeared earlier,
- final-looking statements that are internally coherent.

Scoring:
- 5: No explicit internal mathematical error or self-contradiction is visible.
- 4: Internally coherent, but terse, compressed, or mildly ambiguous.
- 3: Locally questionable; visible ambiguity affects the immediate claim, but no clear error is established.
- 2: Clear local mathematical error, illegal manipulation, or incompatible condition.
- 1: Multiple serious local errors or major internal inconsistency.
- 0: Nonsensical, incoherent, or directly self-contradictory throughout.

Uncertainty policy:
- If the issue is missing context, do not penalize heavily.
- If no explicit local error is visible, prefer 4 or 5.
- Use 0, 1, or 2 only for visible errors or contradictions.

Output only one valid JSON object:
{"score": k}

where k is one integer in {0,1,2,3,4,5}.
"""
        self.system_message_with_reference = """
# Reasoning: High

You are a local mathematical claim-consistency judge.

Your task is to evaluate whether CURRENT_STEP is consistent with one
LOCAL_CLAIM from the same current step.

This is not a reference-free check and not a global proof evaluation.
The LOCAL_CLAIM is a local anchor for consistency checking only.

Do not require the LOCAL_CLAIM to prove the whole CURRENT_STEP.
Do not penalize CURRENT_STEP merely because it contains additional claims,
omits derivations, or depends on previous context.

Judge only whether CURRENT_STEP:
- preserves the meaning of LOCAL_CLAIM,
- keeps the same symbols, conditions, quantifiers, and scope,
- avoids contradicting LOCAL_CLAIM,
- avoids weakening, strengthening, or generalizing LOCAL_CLAIM without support,
- avoids using LOCAL_CLAIM outside its stated domain or case.

Use no original problem statement and no claims from other steps.

Scoring:
- 5: CURRENT_STEP is fully consistent with LOCAL_CLAIM.
- 4: Mostly consistent, with only minor ambiguity or compression.
- 3: Compatible but weakly connected, ambiguous, or hard to verify locally.
- 2: Significant tension, possible scope distortion, or likely misuse.
- 1: Severe inconsistency or major distortion of LOCAL_CLAIM.
- 0: Direct contradiction with LOCAL_CLAIM.

Uncertainty policy:
- If there is no visible contradiction and the issue is merely missing context, prefer 4 or 5.
- If the local relation is unclear but not contradictory, use 3.
- Use 0, 1, or 2 only for visible conflict, misuse, or scope distortion.

Output only one valid JSON object:
{"score": k}

where k is one integer in {0,1,2,3,4,5}.
"""
        # Backward-compatible default. The active message is switched by the two build_user_* methods.
        self.system_message = self.system_message_without_reference
        self.active_system_message = self.system_message_without_reference
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
        self.active_system_message = self.system_message_without_reference
        self.user_message = f"""
## Task
Reference-free internal consistency check.

Evaluate only explicit mathematical correctness and self-consistency inside GEN.
Do not judge whether GEN follows from previous steps or from the original problem.
Do not penalize missing context or omitted derivations.

Penalize only visible issues such as:
- arithmetic/algebraic mistakes,
- illegal operations,
- wrong sign or inequality-direction changes,
- incompatible constraints,
- symbol redefinition conflicts,
- self-contradictions.

If GEN is terse but internally coherent, give a high score.

## GEN
{gen_text}

Output exactly one JSON object: {{"score": k}}
Valid k values: 0, 1, 2, 3, 4, 5.
"""

    def build_user_with_reference(self, gen_text: str, ref_claim_text: str, step_label: str | None = None) -> None:
        self.active_system_message = self.system_message_with_reference
        label = step_label or "local current-step claim"
        self.user_message = f"""
## Task
Local claim-consistency check.

You are given one LOCAL_CLAIM and the full CURRENT_STEP.
Judge whether CURRENT_STEP is internally consistent with this LOCAL_CLAIM.

The LOCAL_CLAIM is an anchor for consistency checking only.
Do not require this single claim to prove the whole CURRENT_STEP.
Do not penalize CURRENT_STEP for containing additional claims, unless those
additional claims contradict, distort, or misuse LOCAL_CLAIM.

Use no original problem statement and no claims from other steps.

Check whether CURRENT_STEP:
- preserves the meaning of LOCAL_CLAIM,
- keeps the same symbols, conditions, quantifiers, and scope,
- avoids contradicting or weakening/strengthening LOCAL_CLAIM,
- avoids using LOCAL_CLAIM outside its stated domain or case.

## CURRENT_STEP_LABEL
{label}

## LOCAL_CLAIM
{ref_claim_text}

## CURRENT_STEP
{gen_text}

Output exactly one JSON object: {{"score": k}}
Valid k values: 0, 1, 2, 3, 4, 5.
"""

        
    def return_prompt(self) -> str:
        return {
            "messages": [
                {"role": "system", "content": getattr(self, "active_system_message", self.system_message)},
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
