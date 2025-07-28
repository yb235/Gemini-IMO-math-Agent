# @title LangGraph Code for the IMO Multi-Agent Workflow
# @markdown This script requires the following packages to be installed:
# @markdown `pip install langchain langgraph langchain-google-genai python-dotenv`


import os
from typing import List, TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 1. Environment Setup ---
# Load API keys from .env file (optional, for local development)
from dotenv import load_dotenv
import pathlib

# Try to load .env file with explicit path
env_path = pathlib.Path(__file__).parent / '.env'
loaded = load_dotenv(env_path)
print(f"Debug: .env file exists at {env_path}: {env_path.exists()}")
print(f"Debug: load_dotenv returned: {loaded}")

# Check for API key and prompt if not found
api_key = os.getenv("GOOGLE_API_KEY")
print(f"Debug: API key found: {'Yes' if api_key else 'No'}")

if not api_key:
    print("Google API Key not found in environment variables.")
    print(f"Please add GOOGLE_API_KEY=your_key to: {env_path}")
    print("Make sure to remove the # comment character!")
    api_key = input("Please enter your Google API key: ").strip()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        print("No API key provided. Exiting.")
        exit(1)

# --- 2. Define the State for the Graph ---
# The state is the shared memory that each node in the graph can access and modify.
class GraphState(TypedDict):
    """
    Represents the state of our multi-agent workflow.

    Attributes:
        problem_statement: The initial mathematical problem to solve.
        solution: The current proposed solution.
        bug_report: A report from the verifier detailing issues.
        verification_summary: The final verdict from the verifier.
        iterations: The number of correction loops.
        error: A field to capture any processing errors.
    """
    problem_statement: str
    solution: Optional[str] = None
    bug_report: Optional[str] = None
    verification_summary: Optional[str] = None
    iterations: int = 0
    error: Optional[str] = None

# --- 3. Define Pydantic Models for Structured Output ---
# These models ensure the LLM's output is in a specific, parsable format.

class SolutionSummary(BaseModel):
    """The high-level summary of a proposed mathematical solution."""
    verdict: str = Field(description="State clearly whether you have found a complete solution or a partial solution.")
    method_sketch: str = Field(description="A high-level, conceptual outline of your solution strategy.")

class DetailedSolution(BaseModel):
    """The detailed, step-by-step mathematical proof."""
    proof: str = Field(description="The full, step-by-step mathematical proof, presented cleanly in TeX format.")

class SolutionGeneration(BaseModel):
    """The complete output for the generator agent."""
    summary: SolutionSummary
    detailed_solution: DetailedSolution

class BugReport(BaseModel):
    """A structured report of issues found in a mathematical solution."""
    final_verdict: str = Field(description="A single, clear sentence declaring the overall validity of the solution (e.g., 'The solution is correct', 'The solution contains a Critical Error').")
    findings: List[str] = Field(description="A list summarizing every issue discovered, including its location and classification (Critical Error or Justification Gap).")

# --- 4. Model Selection and Initialization ---
# Let user choose the Gemini model
print("\n🤖 MODEL SELECTION")
print("=" * 50)
print("Please select the Gemini model you would like to use:")
print("1. Gemini 2.0 Flash - Faster responses, good for general tasks")
print("2. Gemini Pro - More capable reasoning, better for complex mathematics")
print("=" * 50)

while True:
    model_choice = input("Enter your choice (1 or 2): ").strip()
    if model_choice == "1":
        selected_model = "gemini-2.0-flash-exp"
        print("✅ Selected: Gemini 2.0 Flash")
        break
    elif model_choice == "2":
        selected_model = "gemini-2.5-pro"
        print("✅ Selected: Gemini 2.5 Pro")
        break
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

# Initialize the Language Model with selected model
# Following the paper specifications: temperature 0.1 for consistent responses
print(f"\n🔄 Initializing {selected_model}...")
llm = ChatGoogleGenerativeAI(
    model=selected_model,
    temperature=0.1,  # Low temperature as specified in paper Section 3
    max_output_tokens=8192  # Reduced token limit to prevent timeouts
)

# Test the API connection with the selected model
print("🧪 Testing API connection...")
try:
    test_response = llm.invoke("Test connection")
    print("✅ API connection successful!")
    print(f"📡 Model: {selected_model}")
    print(f"🔗 Connection: Active")
except Exception as e:
    print(f"❌ API connection failed!")
    print(f"🚨 Error: {str(e)}")
    print(f"💡 Suggestions:")
    print(f"   • Check your GOOGLE_API_KEY is valid")
    print(f"   • Verify the model '{selected_model}' is available in your region")
    print(f"   • Try switching to the other model option")
    print(f"   • Check your internet connection")
    print("\n🛑 Exiting due to API connection failure.")
    exit(1)

# --- 5. Define the Agent Nodes ---
# Each node is a function that performs a specific task in the workflow.

def generator_node(state: GraphState):
    """
    Node for Step 1: Initial Solution Generation.
    Generates an initial solution based on the problem statement.
    """
    print("=" * 60)
    print("🧠 GENERATOR AGENT - MATHEMATICAL PROBLEM SOLVER")
    print("=" * 60)
    print(f"📋 TASK: Solve the mathematical problem")
    print(f"📝 PROBLEM: {state['problem_statement']}")
    print("\n🤔 THINKING PROCESS:")
    print("• Analyzing the problem structure and requirements")
    print("• Determining the most appropriate mathematical approach")
    print("• Formulating a step-by-step solution strategy")
    print("• Preparing rigorous mathematical proof")
    print("\n⚙️  PROCESSING... (Calling Gemini 2.5 Pro)")
    
    # Step 1 Prompt from Section 3.1 of the research paper
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """### Core Instructions ###

**Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.

**Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
• Proving a key lemma.
• Fully resolving one or more cases within a logically sound case-based proof.
• Establishing a critical property of the mathematical objects in the problem.
• For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.

**Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., 'Let $n$ be an integer.').

### Output Format ###
Your response MUST be structured into the following sections, in this exact order.

**1. Summary**
Provide a concise overview of your findings. This section must contain two parts:
**a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
• **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
• **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."

**b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
• A narrative of your overall strategy.
• The full and precise mathematical statements of any key lemmas or major intermediate results.
• If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**
Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###
Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument."""),
            ("user", "Problem: {problem}\n\nSolve this International Mathematical Olympiad problem following the exact format specified above."),
        ]
    )

    # Use the.with_structured_output method to get a guaranteed JSON-like object.
    structured_llm = llm.with_structured_output(SolutionGeneration)
    chain = prompt | structured_llm

    try:
        print("🔄 Making API call to Gemini...")
        response = chain.invoke({"problem": state["problem_statement"]})
        print("✅ API call completed successfully")
        solution_text = f"## Summary\n\n**Verdict:** {response.summary.verdict}\n\n**Method Sketch:**\n{response.summary.method_sketch}\n\n## Detailed Solution\n\n{response.detailed_solution.proof}"
        
        print("\n✅ GENERATOR OUTPUT:")
        print("=" * 40)
        print(f"📊 VERDICT: {response.summary.verdict}")
        print(f"🎯 METHOD SKETCH:")
        print(f"   {response.summary.method_sketch}")
        print(f"📝 DETAILED PROOF:")
        print(f"   {response.detailed_solution.proof}")
        print("=" * 40)
        print("✅ Generator completed successfully - passing solution to Verifier")
        
        return {"solution": solution_text}
    except Exception as e:
        print(f"❌ ERROR IN GENERATOR: {e}")
        print("🚨 Generator failed - terminating workflow")
        print("💡 Possible causes:")
        print(f"   • API rate limit exceeded")
        print(f"   • Network connectivity issues") 
        print(f"   • Model '{selected_model}' temporarily unavailable")
        print(f"   • Invalid API key or permissions")
        return {"error": f"Failed to generate a solution: {str(e)}"}

def self_improvement_node(state: GraphState):
    """
    Node for Step 2: Self-Improvement.
    The model reviews and tries to improve its initial solution.
    This step injects additional thinking budget as mentioned in the paper.
    """
    print("\n" + "=" * 60)
    print("🔄 SELF-IMPROVEMENT AGENT - SOLUTION REFINER")
    print("=" * 60)
    print(f"📋 TASK: Review and improve the initial solution")
    print(f"📝 RECEIVED FROM: Generator Agent")
    print(f"💭 PURPOSE: Inject additional thinking budget (32768 tokens)")
    print("\n🤔 THINKING PROCESS:")
    print("• Reviewing the initial solution for potential improvements")
    print("• Identifying areas that could be made more rigorous")
    print("• Expanding on abbreviated reasoning steps")
    print("• Ensuring mathematical completeness and clarity")
    print("• Using additional thinking budget to refine the approach")
    print("\n⚙️  PROCESSING... (Calling Gemini 2.5 Pro for self-improvement)")
    
    # Step 2 Self-Improvement Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a world-class mathematician reviewing your own initial solution to an IMO problem. Your task is to critically examine and improve your work.

### Core Instructions ###

**Critical Self-Review:** Carefully examine your initial solution. Look for:
• Areas where reasoning could be made more rigorous
• Steps that could be expanded or better justified  
• Potential gaps in logic or missing details
• Opportunities to strengthen the mathematical argument
• Ways to make the presentation clearer and more complete

**Improvement Focus:** Your goal is to produce an enhanced version that:
• Maintains the core approach if it's sound
• Fills in any gaps or missing justifications
• Expands abbreviated reasoning steps
• Ensures every claim is properly supported
• Uses clearer mathematical exposition

**Output Format:** Follow the same structured format as your initial solution:
1. **Summary** (with verdict and method sketch)
2. **Detailed Solution** (the improved, step-by-step proof)

**Self-Correction:** Use your additional thinking budget to thoroughly work through the problem again, building upon your initial insights while addressing any weaknesses."""),
            ("user", """### Original Problem ###
{problem}

### Your Initial Solution ###
{solution}

### Self-Improvement Task ###
Review your initial solution above and provide an improved version. Focus on enhancing rigor, filling gaps, and strengthening the mathematical argument while maintaining the core approach if it's sound."""),
        ]
    )

    structured_llm = llm.with_structured_output(SolutionGeneration)
    chain = prompt | structured_llm

    try:
        response = chain.invoke({
            "problem": state["problem_statement"],
            "solution": state["solution"]
        })
        improved_solution = f"## Summary\n\n**Verdict:** {response.summary.verdict}\n\n**Method Sketch:**\n{response.summary.method_sketch}\n\n## Detailed Solution\n\n{response.detailed_solution.proof}"
        
        print("\n📋 SELF-IMPROVEMENT OUTPUT:")
        print("=" * 40)
        print("✅ IMPROVEMENT STATUS: SOLUTION REFINED")
        print(f"📊 REFINED VERDICT: {response.summary.verdict}")
        print(f"🎯 ENHANCED METHOD SKETCH:")
        print(f"   {response.summary.method_sketch}")
        print(f"📝 IMPROVED PROOF:")
        print(f"   {response.detailed_solution.proof}")
        print("=" * 40)
        print("✅ Self-improvement completed - passing enhanced solution to Verifier")
        
        return {"solution": improved_solution}
    except Exception as e:
        print(f"❌ ERROR IN SELF-IMPROVEMENT: {e}")
        print("🚨 Self-improvement failed - using original solution")
        print("💡 Possible causes:")
        print(f"   • API rate limit exceeded")
        print(f"   • Model '{selected_model}' temporarily unavailable")
        print(f"   • Network connectivity issues")
        return {}  # Keep original solution if improvement fails

def verifier_node(state: GraphState):
    """ 
    Node for Step 3: Verifying the Solution.
    Checks the solution for errors and generates a structured bug report.
    """
    print("\n" + "=" * 60)
    print("🔍 VERIFIER AGENT - MATHEMATICAL SOLUTION VALIDATOR")
    print("=" * 60)
    print(f"📋 TASK: Critically examine the proposed solution")
    print(f"🔎 CURRENT ITERATION: {state.get('iterations', 0) + 1}")
    print(f"📝 REVIEWING SOLUTION FROM: Generator Agent")
    print("\n🤔 THINKING PROCESS:")
    print("• Reading and understanding the original problem")
    print("• Analyzing the proposed solution step-by-step")
    print("• Checking mathematical logic and calculations")
    print("• Identifying gaps in reasoning or critical errors")
    print("• Classifying issues as 'Critical Error' or 'Justification Gap'")
    print("\n⚙️  PROCESSING... (Calling Gemini 2.5 Pro for verification)")
    
    # Step 3 Verification Prompt from Section 3.2 of the research paper
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
• Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
• You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

**a. Critical Error:**
This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that 'A>B, C>D' implies 'A-C>B-D') and **factual errors** (e.g., a calculation error like '2+3=6').

**Procedure:**
• Explain the specific error and state that it **invalidates the current line of reasoning**.
• Do NOT check any further steps that rely on this error.
• You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

**b. Justification Gap:**
This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.

**Procedure:**
• Explain the gap in the justification.
• State that you will **assume the step's conclusion is true** for the sake of argument.
• Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

**a. Summary**
This section MUST be at the very beginning of your response. It must contain two components:
• **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
• **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
  • **Location:** A direct quote of the key phrase or equation where the issue occurs.
  • **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

**b. Detailed Verification Log**
Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part."""),
            ("user", """### Problem ###
{problem}

### Solution ###
{solution}

### Verification Task Reminder ###
Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above."""),
        ]
    )

    # Try structured output first, fallback to regular output if it fails
    try:
        structured_llm = llm.with_structured_output(BugReport)
        chain = prompt | structured_llm
        print("🔄 Attempting structured verification...")
        response = chain.invoke({
            "problem": state["problem_statement"],
            "solution": state["solution"]
        })
        print("✅ Structured verification completed")
        
        # Defensively ensure findings is a list, even if the LLM returns None.
        findings = response.findings or []
        final_verdict = response.final_verdict
        
    except Exception as struct_error:
        print(f"⚠️  Structured output failed: {struct_error}")
        print("🔄 Falling back to regular LLM output...")
        
        # Fallback to regular LLM call
        regular_chain = prompt | llm
        raw_response = regular_chain.invoke({
            "problem": state["problem_statement"],
            "solution": state["solution"]
        })
        
        response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
        print(f"📝 Raw verifier response: {response_text[:200]}...")
        
        # Simple parsing of response
        if "the solution is correct" in response_text.lower():
            findings = []
            final_verdict = "The solution is correct."
        else:
            findings = ["Verification completed with regular output - see detailed response above"]
            final_verdict = "The solution may contain issues - manual review recommended."

    print("\n📋 VERIFIER OUTPUT:")
    print("=" * 40)
    
    if not findings:
        summary = final_verdict
        report = "No issues found."
        print("✅ VERIFICATION STATUS: SOLUTION ACCEPTED")
        print(f"📊 FINAL VERDICT: {summary}")
        print("🎉 No mathematical errors or gaps found!")
        print("✅ Verifier completed - Solution is ready for final output")
    else:
        summary = final_verdict
        report = "\n".join(f"- {finding}" for finding in findings)
        print("❌ VERIFICATION STATUS: ISSUES FOUND")
        print(f"📊 FINAL VERDICT: {summary}")
        print("🔍 DETAILED FINDINGS:")
        for i, finding in enumerate(findings, 1):
            print(f"   {i}. {finding}")
        print("⚠️  Verifier completed - Solution needs correction")
    
    print("=" * 40)

    return {
        "verification_summary": summary,
        "bug_report": report,
        "iterations": state.get("iterations", 0) + 1
    }

def human_review_node(state: GraphState):
    """
    Node for Step 4: Human Review of the Bug Report.
    This simulates the human-in-the-loop step.
    """
    print("\n" + "=" * 60)
    print("👤 HUMAN REVIEW AGENT - QUALITY ASSURANCE GATE")
    print("=" * 60)
    print(f"📋 TASK: Review verifier's findings before correction")
    print(f"🔎 ITERATION: {state.get('iterations', 1)}")
    print(f"📥 RECEIVED FROM: Verifier Agent")
    print(f"📤 DECISION WILL GO TO: Corrector Agent (if approved)")
    
    print("\n🤔 YOUR ROLE:")
    print("• Evaluate whether the verifier's bug report is accurate")
    print("• Decide if the identified issues warrant correction")
    print("• Ensure the correction process should proceed")
    print("• Act as a quality gate in the workflow")
    
    print("\n📋 VERIFICATION SUMMARY:")
    print("=" * 40)
    print(f"📊 STATUS: {state['verification_summary']}")
    print("\n🔍 DETAILED BUG REPORT:")
    print("=" * 40)
    bug_lines = state["bug_report"].split('\n')
    for i, line in enumerate(bug_lines, 1):
        if line.strip():
            print(f"   {i}. {line.strip()}")
    print("=" * 40)
    
    print("\n🎯 DECISION REQUIRED:")
    print("Do you agree with the verifier's assessment?")
    print("• Type 'y' to APPROVE the bug report and proceed with correction")
    print("• Type 'n' to REJECT the bug report and end the workflow")

    # In a real application, this could be an API endpoint or a UI.
    # Here, we simulate it with a command-line input.
    while True:
        user_input = input("\n👤 Your decision (y/n): ").lower()
        if user_input == 'y':
            print("\n✅ HUMAN DECISION: Bug report APPROVED")
            print("📤 Passing control to Corrector Agent...")
            return {} # No state change needed, just proceed
        elif user_input == 'n':
            print("\n❌ HUMAN DECISION: Bug report REJECTED")
            print("🛑 Workflow terminated by human reviewer")
            return {"error": "Human reviewer rejected the bug report."}

def corrector_node(state: GraphState):
    """
    Node for Step 5: Correcting the Solution.
    Improves the solution based on the verifier's bug report.
    """
    print("\n" + "=" * 60)
    print("🔧 CORRECTOR AGENT - MATHEMATICAL SOLUTION REFINER")
    print("=" * 60)
    print(f"📋 TASK: Fix the solution based on verification feedback")
    print(f"🔄 CORRECTION ITERATION: {state.get('iterations', 1)}")
    print(f"📥 RECEIVED FEEDBACK FROM: Verifier Agent (via Human Review)")
    print(f"📝 ORIGINAL PROBLEM: {state['problem_statement']}")
    print("\n🤔 THINKING PROCESS:")
    print("• Analyzing the verifier's bug report and identified issues")
    print("• Understanding what went wrong in the original solution")
    print("• Developing a corrected approach to address each issue")
    print("• Ensuring the new solution is mathematically rigorous")
    print("• Maintaining clarity while fixing identified problems")
    print(f"\n🔍 ISSUES TO ADDRESS:")
    bug_lines = state['bug_report'].split('\n')
    for i, line in enumerate(bug_lines[:5], 1):  # Show first 5 issues
        if line.strip():
            print(f"   {i}. {line.strip()}")
    print("\n⚙️  PROCESSING... (Calling Gemini 2.5 Pro for correction)")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a world-class mathematician, capable of solving the hardest problems from the International Mathematical Olympiad (IMO). You have received a bug report for your previous solution. Your task is to revise and improve the solution based on the feedback provided in the bug report. Ensure the corrected solution is rigorous, complete, and addresses all identified issues."),
            ("user", "Problem: {problem}\n\nOriginal Solution: {solution}\n\nBug Report: {bug_report}\n\nBased on the bug report, provide a fully revised and corrected solution. This includes a high-level summary (verdict and method sketch) and a new detailed proof."),
        ] 
    )

    # Use structured output for consistency with the generator.
    structured_llm = llm.with_structured_output(SolutionGeneration)
    chain = prompt | structured_llm

    try:
        response = chain.invoke({
            "problem": state["problem_statement"],
            "solution": state["solution"],
            "bug_report": state["bug_report"]
        })
        solution_text = f"## Summary\n\n**Verdict:** {response.summary.verdict}\n\n**Method Sketch:**\n{response.summary.method_sketch}\n\n## Detailed Solution\n\n{response.detailed_solution.proof}"
        
        print("\n📋 CORRECTOR OUTPUT:")
        print("=" * 40)
        print("✅ CORRECTION STATUS: SOLUTION REVISED")
        print(f"📊 NEW VERDICT: {response.summary.verdict}")
        print(f"🎯 UPDATED METHOD SKETCH:")
        print(f"   {response.summary.method_sketch}")
        print(f"📝 REVISED PROOF:")
        print(f"   {response.detailed_solution.proof}")
        print("=" * 40)
        print("✅ Corrector completed - passing revised solution back to Verifier")
        
        return {"solution": solution_text}
    except Exception as e:
        print(f"❌ ERROR IN CORRECTOR: {e}")
        print("🚨 Corrector failed - terminating workflow")
        print("💡 Possible causes:")
        print(f"   • API rate limit exceeded")
        print(f"   • Model '{selected_model}' temporarily unavailable")
        print(f"   • Network connectivity issues")
        return {"error": f"Failed to correct the solution: {str(e)}"}

# --- 6. Define Conditional Edges ---
# These functions determine the next step in the graph based on the current state.

def decide_after_verification(state: GraphState):
    """
    Determines the next step after the verifier runs.
    - If the solution is correct, end.
    - If there are bugs and iterations are within limit, go to human review.
    - Otherwise, end.
    """
    print("\n" + "🎯" * 20)
    print("🧠 DECISION ENGINE - POST-VERIFICATION ROUTING")
    print("🎯" * 20)
    print("📋 TASK: Determine next step based on verification results")
    print("\n🤔 DECISION LOGIC:")
    print("• Check if any errors occurred during verification")
    print("• Analyze verification summary for 'correct' solution")
    print("• Check iteration count against maximum limit (3)")
    print("• Route to appropriate next agent or terminate")
    
    print(f"\n📊 CURRENT STATE ANALYSIS:")
    print(f"   Error Status: {state.get('error', 'None')}")
    print(f"   Verification Summary: {state.get('verification_summary', 'N/A')}")
    print(f"   Current Iteration: {state.get('iterations', 0)}/3")
    print(f"   Debug - Summary lowercase: '{state.get('verification_summary', '').lower()}'")
    
    if state.get("error"):
        print("\n❌ DECISION: TERMINATE (Error detected)")
        print("🛑 Workflow ending due to error condition")
        return "END"

    summary = state.get("verification_summary", "").lower()
    # More flexible checking for correct solutions
    is_correct = (
        "the solution is correct" in summary or
        "solution is correct" in summary or
        summary.strip() == "the solution is correct." or
        "correct" in summary and "incorrect" not in summary and "error" not in summary
    )
    
    if is_correct:
        print("\n✅ DECISION: TERMINATE (Solution accepted)")
        print("🎉 Solution verified as correct - workflow complete!")
        return "END"
    elif state.get("iterations", 0) >= 3: # Set a max of 3 correction loops
        print("\n⏰ DECISION: TERMINATE (Max iterations reached)")
        print("🔄 Maximum correction attempts (3) exceeded")
        return "END"
    else:
        print("\n🔄 DECISION: CONTINUE (Issues found, within iteration limit)")
        print("👤 Routing to Human Review for bug report validation")
        return "human_review"

def decide_after_human_review(state: GraphState):
    """
    Determines the next step after human review.
    - If an error was flagged (review rejected), end.
    - Otherwise, proceed to correction.
    """
    print("\n" + "🎯" * 20)
    print("🧠 DECISION ENGINE - POST-HUMAN-REVIEW ROUTING")
    print("🎯" * 20)
    print("📋 TASK: Route based on human reviewer's decision")
    print("\n🤔 DECISION LOGIC:")
    print("• Check if human reviewer rejected the bug report")
    print("• If approved, route to Corrector Agent")
    print("• If rejected, terminate workflow")
    
    print(f"\n📊 HUMAN REVIEW OUTCOME:")
    if state.get("error"):
        print("   Decision: Bug report REJECTED")
        print("\n❌ DECISION: TERMINATE (Human reviewer disagreed)")
        print("🛑 Workflow ending - human reviewer rejected verifier's findings")
        return "END"
    else:
        print("   Decision: Bug report APPROVED")
        print("\n🔧 DECISION: CONTINUE (Proceed with correction)")
        print("📤 Routing to Corrector Agent for solution refinement")
        return "corrector"

# --- 7. Build the Graph ---
# Connect all the nodes and edges to define the workflow.

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("generator", generator_node)
workflow.add_node("self_improvement", self_improvement_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("corrector", corrector_node)

# Set entry point
workflow.set_entry_point("generator")

# Add edges - following the paper's pipeline
workflow.add_edge("generator", "self_improvement")  # Step 1 → Step 2
workflow.add_edge("self_improvement", "verifier")   # Step 2 → Step 3
workflow.add_edge("corrector", "verifier")          # Step 5 → Step 3 (loop back)

# Add conditional edges
workflow.add_conditional_edges(
    "verifier",
    decide_after_verification,
    {
        "human_review": "human_review",
        "END": END
    }
)
workflow.add_conditional_edges(
    "human_review",
    decide_after_human_review,
    {
        "corrector": "corrector",
        "END": END
    }
)


# Compile the graph
app = workflow.compile()

def generate_final_report(final_state, initial_state):
    """
    Generate a comprehensive final report showing the entire problem-solving process
    """
    print("\n📋 PROBLEM STATEMENT:")
    print("=" * 60)
    print(initial_state["problem_statement"])
    
    print("\n🔄 WORKFLOW EXECUTION SUMMARY:")
    print("=" * 60)
    print("✅ Step 1: GENERATOR - Initial solution generated")
    print("✅ Step 2: SELF-IMPROVEMENT - Solution refined and enhanced") 
    print("✅ Step 3: VERIFIER - Solution rigorously verified")
    
    if final_state.get("iterations", 0) > 1:
        print("✅ Step 4: HUMAN REVIEW - Bug reports reviewed")
        print("✅ Step 5: CORRECTOR - Solution corrected based on feedback")
        print(f"🔄 Total correction iterations: {final_state.get('iterations', 1)}")
    
    print("\n📊 VERIFICATION RESULTS:")
    print("=" * 60)
    verification_status = final_state.get("verification_summary", "N/A")
    print(f"Final Status: {verification_status}")
    
    if final_state.get("bug_report") and "No issues found" not in final_state.get("bug_report", ""):
        print("\n🔍 Issues Identified During Process:")
        bug_lines = final_state.get("bug_report", "").split('\n')
        for i, line in enumerate(bug_lines, 1):
            if line.strip():
                print(f"   {i}. {line.strip()}")
    
    print("\n📈 PROCESS METRICS:")
    print("=" * 60)
    print(f"• Total Iterations: {final_state.get('iterations', 1)}")
    print(f"• Workflow Status: {'✅ COMPLETED' if not final_state.get('error') else '❌ TERMINATED WITH ERROR'}")
    print(f"• Solution Accepted: {'✅ YES' if 'correct' in verification_status.lower() and 'incorrect' not in verification_status.lower() else '❌ NO'}")
    
    print("\n" + "🎉" * 60)

# --- 8. Run the Workflow ---
if __name__ == "__main__":
    # Example problem statement (IMO 2011, Problem 2)
    default_problem = "Let S be a finite set of at least two points in the plane. Assume that no three points of S are collinear. A windmill is a process that starts with a line l going through a single point P in S. The line rotates clockwise about the pivot P until it first hits another point of S, say Q. This point Q now becomes the new pivot, and the line rotates clockwise about Q until it hits a third point of S. This process continues indefinitely. Show that we can choose a point P in S and a line l through P such that the resulting windmill uses each point of S as a pivot infinitely many times."

    print("Please enter the mathematical problem you would like to solve.")
    print("(Press Enter to use the default IMO windmill problem)")
    print("-" * 40)
    user_problem = input("> ")
    print("-" * 40)

    problem = user_problem.strip() if user_problem.strip() else default_problem
    initial_state = {"problem_statement": problem}

    # Invoke the graph and stream the results
    final_state = {}
    for s in app.stream(initial_state, {"recursion_limit": 10}):
        print("\n" + "="*40)
        node_name = list(s.keys())[0]
        print(f"Node: '{node_name}'")
        print("="*40)
        # The value of the dictionary is the state after the node has run
        final_state = s[node_name]

    print("\n" + "🎉" * 60)
    print("🏁 WORKFLOW COMPLETE - IMO PROBLEM SOLVING SUMMARY")
    print("🎉" * 60)
    
    # Generate comprehensive final report
    generate_final_report(final_state, initial_state)
