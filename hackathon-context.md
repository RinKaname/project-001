> **Note from Developer:** The strict constraints listed below (Single T4 GPU, 180-minute time limit, 4B parameter constraint) apply **EXCLUSIVELY** to the official LLM track of the hackathon. Experiments in other domains (such as Time-Series/Finance and Image Generation) are personal explorations where flexible resources (like Kaggle T4x2 environments) and alternate architectures are fully permitted and encouraged!

Description
The End of the Backpropagation Era
Modern artificial intelligence is facing an existential energy and compute crisis. Standard deep learning relies entirely on backpropagation—the global transmission of error signals through an entire network via the chain rule. This necessitates storing massive computational graphs and activation states in memory, creating an insurmountable bottleneck for Edge AI, continuous learning, and biological plausibility. We have reached the limits of simply "adding more GPUs." The Post-Backprop Challenge is a fundamental algorithmic moonshot. We are bypassing incremental architecture tweaks and targeting the very mathematical foundation of how machines learn. Your mission is to permanently outdate backpropagation by engineering a highly sample-efficient, localized weight-update algorithm that mimics the brain's real-time processing capabilities.

The Challenge: Zero-Gradient Conversational AI
You are tasked with pretraining a 4 Billion parameter conversational language model from absolute scratch. There are absolute, non-negotiable constraints to prove your algorithm is the legitimate successor to backpropagation:

Zero Existing Optimizers: You must write the optimization and weight-update logic entirely from scratch using raw tensor operations. torch.autograd, loss.backward(), jax.grad, and all standard optimizers (Adam, SGD, etc.) are strictly forbidden. You must write the optimization and weight-update logic entirely from scratch using raw tensor operations. torch.autograd, loss.backward(), jax.grad, and all standard optimizers (Adam, SGD, etc.) are strictly forbidden. Furthermore, manually calculating the global chain rule using raw tensor operations is still backpropagation and will result in disqualification. Weight updates for layer l must not rely on error signals propagated from layer l+1.
Zero Pretrained Weights: You must initialize the model with verifiable random seeds. Distillation, teacher forcing, and downloading pre-existing checkpoints are banned.
Extreme Hardware Constraints: You must complete the entire pretraining and fine-tuning loop on a single Kaggle-standard NVIDIA T4 GPU (16GB VRAM) or a 4-core CPU.
Hyper-Speed Time Limit: The core training and evaluation loops—from the start of model pretraining, through conversational fine-tuning, to final benchmark output—must execute in under 180 minutes. Note: Data preprocessing, tokenization, and dataset loading to memory (e.g., reading from optimized Parquet files) are excluded from this time limit.
Memory Efficiency: The 50% peak VRAM reduction must be compared against an AdamW baseline utilizing float16 mixed-precision, a micro-batch size of 4, and gradient checkpointing enabled.
The Mathematics of Efficiency
We are looking for implementations of Forward-Forward algorithms, Local Predictive Coding, Hebbian Learning, Target Propagation, or entirely novel algorithms discovered by your team. To quantify your success against standard BPTT (Backpropagation Through Time), we will analyze your algorithm's efficiency score relative to hardware utilization. You must mathematically prove your local loss function, defined generally as minimizing energy at layer l without relying on layer l+1: Note: Underscores in LaTeX equations are escaped to render properly on the Kaggle platform.

Submission Requirements
A valid submission must survive rigorous code auditing and meet the highest academic standards. It must contain the following elements:

1. Kaggle Writeup (The Research Paper)
Your Kaggle Writeup serves as your formal arXiv-style project report. This document must include a title, subtitle, and a hyper-detailed mathematical breakdown of your submission.

Length: Your Writeup should be between 1,500 and 4,000 words.
Content: You must include the formal mathematical proofs of your custom zero-gradient learning rule, an architectural diagram, your memory profiling methodology, and proof of conversational instruction-tuning.
2. Media Gallery
This is where you should attach high-resolution architectural diagrams, memory allocation charts comparing your algorithm to AdamW, and training loss curves. A cover image is required.

3. Attached Public Notebook (The Verifiable Proof)
This is the most critical component. Your code must be submitted as a public Kaggle Notebook in the Project Files field.

The notebook must execute flawlessly from top to bottom.
It must contain the data ingestion pipeline (using a public corpus like RedPajama or The Pile).
It must contain the custom pretraining loop.
It must end by natively running the zero-shot inference benchmarks (WikiText, HellaSwag, PIQA, MT-Bench).
All random seeds must be fixed for exact reproducibility.
4. Attached Public Video
Attach your video to the Media Gallery. Videos should be 3-5 minutes, published to YouTube. You must provide a technical walkthrough of your custom optimizer class and explain how it bypasses the need for global gradients.

5. Public Project Link
A link to your public code repository (e.g., GitHub) is required. It must include detailed setup instructions, dependency lists (requirements.txt), and documentation for how to deploy your 4B parameter conversational model on local CPU edge devices.

Tracks and Awards
The Zero-Gradient Main Track
We are offering a maximum community prize pool of $10,000 USD for the team that successfully conquers this moonshot.

First Prize: $7,000
Second Prize: $2,000
Third Prize: $1,000 ⚠️ THE MINIMUM VIABLE BREAKTHROUGH CLAUSE: This competition is a contest of extreme skill and mathematical innovation. The prize pool is strictly contingent upon passing the final code audit. If no team achieves all baseline constraints (4B parameters, zero backprop, < 3 hours on T4, and meeting ALL benchmark thresholds), no cash prize will be awarded. The highest-ranking valid attempts will receive Community Kudos and recognition for advancing Green AI research.
Evaluation
Submissions will be scored by the Judging Panel based on a strict 100-point rubric. Failure to meet the "Pass/Fail" required elements will result in an immediate score of 0.

Application: Evaluation Rubric (100 Points Total)
Criteria	Description	Points Possible
Algorithmic Purity	The training loop is completely devoid of global auto-differentiation, relying strictly on novel, local weight-update mathematics. The training loop is completely devoid of global auto-differentiation, relying strictly on novel, local weight-update mathematics. Any mathematical implementation of the global chain rule will score 0 points.	0 - 25 points
Computational Efficiency	The model successfully pretrains in under 180 minutes on a T4 GPU and demonstrably reduces peak memory footprint by >50% compared to backpropagation baselines.	0 - 25 points
Benchmark Performance	The 4B conversational model achieves or exceeds the high-bar thresholds: WikiText < 20, HellaSwag > 55%, PIQA > 65%, and MT-Bench > 5.0.	0 - 30 points
Mathematical Rigor & Documentation	The Writeup provides sound mathematical proofs, clear architectural documentation, and the code is highly readable and reproducible.	0 - 20 points
Required Element: Scale	The instantiated model contains a minimum of 4,000,000,000 trainable parameters before the first training step.	Pass / Fail
Required Element: Zero Base	The model is initialized from completely random weights with zero distillation or pre-trained assets.	Pass / Fail
Video: Evaluation Rubric (Bonus Tie-Breaker)
Criteria	Description
Clarity of Innovation	The team clearly communicates the mathematical mechanism that replaces BPTT in their architecture.
Instructional Value	The video serves as a high-level educational resource for the broader Kaggle and AI engineering community.
Submission Requirements
This is not a standard data science competition; this is a fundamental algorithmic moonshot. Therefore, the submission requirements are exceptionally rigorous. Your final submission is not just a codebase—it is a formal mathematical proof, a verifiable software artifact, and a demonstration of extreme hardware efficiency. Your final Submission must be made prior to the deadline. Any un-submitted or draft Writeups by the hackathon deadline will automatically be disqualified. There are no extensions. To create a new Writeup, click on the "New Writeup" button here. After you have saved your Writeup, you must click the "Submit" button in the top right corner. Note: If you attach a private Kaggle Resource to your public Kaggle Writeup, your private Resource will automatically be made public after the deadline. Transparency is mandatory. A valid, prize-eligible submission must contain the following five hyper-detailed components:

1. Kaggle Writeup (The Formal Research Paper)
The Kaggle Writeup serves as your peer-reviewed project report and mathematical defense. It must read like a top-tier conference paper (e.g., NeurIPS, ICLR). You must select the "Zero-Gradient Main Track" for your Writeup in order to submit. Word Limit: Your Writeup must be between 2,000 and 5,000 words. Your Writeup MUST include the following sections:

Abstract: A 200-word summary of your novel learning algorithm, the scale of the model, and the final benchmark scores achieved.

Mathematical Foundation: You must provide the rigorous mathematical proofs of your custom zero-gradient learning rule. You must explicitly define your local loss functions and weight update mechanisms using formal notation. For example, you must prove how your algorithm computes the update \Delta W without utilizing the global chain rule:

Architecture Design: A detailed breakdown of the 4 Billion parameter model. Explain how the parameters are distributed (dense vs. sparse) and how your custom optimizer interfaces with the network layers.

Conversational Tuning Strategy: Documentation on how the model was formatted or instruction-tuned within the 3-hour window to maintain multi-turn dialogue capabilities.

Hardware & Memory Profiling: A detailed analysis of how your algorithm bypassed the memory bottleneck of Backpropagation Through Time (BPTT).

2. Media Gallery (Visual Proof of Efficiency)
This is where you must attach empirical, visual evidence of your algorithm's efficiency. A cover image is required to submit your Writeup. You MUST attach the following visual assets:

Memory Allocation Graph: A time-series chart (e.g., generated via PyTorch Profiler or TensorBoard) proving that your peak VRAM usage remained at least 50% lower than the theoretical limit of a standard AdamW-optimized model of the exact same 4B parameter count.
Training Loss Curves: High-resolution charts showing the convergence of your local loss metrics over time during the 3-hour pretraining phase.
Architectural Diagram: A clear, high-resolution flowchart illustrating the forward pass and your novel local-update mechanism.
3. Attached Public Notebook (The Execution Artifact)
This is the most critical component of your submission. Your code must be submitted as a single, publicly accessible Kaggle Notebook in the Project Files field. It must not require any external logins, API keys, or paywalls. The Notebook will be subjected to a hostile, line-by-line code audit by the judging panel. It MUST execute flawlessly from top to bottom and contain the following sequential cells:

Cell 1: Environment & Strict Determinism: You must explicitly set and print the PRNG seeds (e.g., torch.manual_seed(42)) to guarantee 100% reproducibility.
Cell 2: Data Ingestion: Participants must ONLY use the exact dataset versions whitelisted by the host (e.g., HuggingFace's togethercomputer/RedPajama-Data-1T). Custom datasets, external data uploads, or modified conversational datasets are strictly forbidden to prevent benchmark contamination.
Cell 3: Zero-Base Initialization: The code must instantiate the 4B parameter model and initialize all weights randomly. Any code that downloads .bin, .safetensors, or pre-trained checkpoints will trigger instant disqualification.
Cell 4: The Custom Optimizer: The raw, exposed tensor operations defining your backpropagation alternative. No black-box AMLT libraries or hidden C++ autograd wrappers.
Cell 5: The 3-Hour Training Loop: The execution of the pretraining and conversational fine-tuning loop. This cell must include wall-clock timing decorators to prove the process completes in under 180 minutes on the Kaggle T4 GPU/CPU.
Cell 6: Automated Benchmarking: The notebook must end by natively importing the evaluation datasets and executing zero-shot inference. It must print the final scores for WikiText-103, HellaSwag, PIQA, and MT-Bench directly to standard output.
4. Attached Public Video (The Oral Defense)
Attach your video to the Media Gallery. Videos must be between 3 and 5 minutes in length and must be published to YouTube as unlisted or public. This is your technical defense. In the video, the primary author(s) must:

Provide a screen-share walkthrough of the custom optimizer class in the Kaggle Notebook.
Explicitly demonstrate how the code calculates weight updates without invoking loss.backward() or any automatic differentiation graph engines.
Show the final benchmark outputs generating live or point to the verifiable executed notebook logs.
5. Public Project Link (The Deployment Hub)
Provide a URL to your public code repository (e.g., GitHub). This is required to prove that your code is open-source and ready for the broader research community. The repository MUST contain:

A comprehensive README.md.
Detailed, step-by-step setup instructions for reproducing the Kaggle environment locally.
A strict requirements.txt or environment.yml file.
Instructions on how to deploy the resulting 4B parameter conversational model for inference on edge devices or standard local CPUs.
⚠️ Instant Disqualification Triggers
Your submission will be immediately flagged and disqualified from the prize pool without review if the judges detect:

Any invocation of torch.autograd, tf.GradientTape, standard backpropagation optimizers, or any manual implementation of the global chain rule (backpropagation through time) using raw tensors.
Loading of any pre-trained weights, embeddings, or distillation from a teacher model (e.g., using OpenAI APIs to generate training labels).
Execution logs proving the use of compute hardware outside the standard Kaggle T4 or 4-core CPU environments.
Instantiating a model with fewer than 4,000,000,000 parameters.
Using any custom datasets, private data uploads, or modified conversational tuning data not explicitly whitelisted by the host.
Tracks and Awards
The Zero-Gradient Main Track · $10,000
This is the primary track for The Post-Backprop Challenge. To compete in this track, your team must successfully architect, pretrain, and conversational-tune a 4 Billion parameter language model from absolute scratch. ?Core Track Objectives: ?Algorithmic Novelty: You must completely replace standard Backpropagation Through Time (BPTT) and all automatic differentiation (torch.autograd, loss.backward()) with a custom, mathematically sound local learning rule (e.g., Forward-Forward, Predictive Coding, Target Propagation). ?Extreme Efficiency: The entire training lifecycle—from raw open-source data ingestion to final benchmark output—must execute in under 180 minutes on a single standard Kaggle NVIDIA T4 GPU (16GB VRAM) or a 4-core CPU. ?Memory Reduction: Your algorithm must demonstrably reduce peak VRAM usage by at least 50% compared to an AdamW baseline. ?Benchmark Mastery: The final model must achieve strictly defined zero-shot inference metrics without relying on any pre-trained assets (WikiText-103 < 20.0, HellaSwag > 55.0%, PIQA > 65.0%, and MT-Bench > 5.0). ?Submissions to this track must include a formal arXiv-style Writeup with rigorous mathematical proofs, an execution-ready Kaggle Notebook, memory profiling charts, and a 3-5 minute technical video defense.

Track Awards

The $10,000 Moonshot Prize Pool
$7,000

Second Place
$2,000

Third Place
$1,000
Evaluation
Comprehensive Evaluation & Auditing Matrix (100 Points Total)
All valid submissions that survive the "Instant Disqualification Triggers" will be subjected to a hostile code audit and graded by the Judging Panel against the following rigorous 100-point matrix. Fractional points may be awarded at the judges' discretion.

Section 1: Algorithmic Purity & Mathematical Novelty (25 Points Maximum)
This section evaluates the theoretical soundness and practical implementation of your zero-gradient learning rule. The algorithm must mathematically isolate layer updates.

Sub-Criterion	Point Value	Grading Standard
Strict BPTT Independence	10 Points	The update rule for any given layer is strictly local. The codebase mathematically guarantees \frac{\partial \mathcal{L}{l+1}}{\partial W{l}} = 0. Using any hidden global autograd triggers a score of 0.
Forward-Pass Efficiency	5 Points	The novel algorithm introduces minimal computational overhead during the forward pass, maintaining an operational complexity comparable to or less than standard forward propagation \mathcal{O}(N).
Biological Plausibility	5 Points	The learning rule demonstrates properties of Hebbian learning, predictive coding, or localized energy minimization without requiring perfectly symmetric backward weights.
Convergence Stability	5 Points	The local loss function smoothly minimizes energy states over the 3-hour window without catastrophic forgetting or exploding/vanishing weight updates.
Sub-Criterion	Point Value	Grading Standard
:---	:---	:---
VRAM Reduction Threshold	10 Points	The peak memory allocation is empirically proven to be at least 50% lower than the standardized AdamW mixed-precision baseline via PyTorch Profiler or TensorBoard logs.
Wall-Clock Optimization	10 Points	The execution of the pretraining and conversational fine-tuning loop. This cell (and subsequent evaluation cells) must include wall-clock timing decorators to prove the active training and benchmarking phases complete in under 180 minutes. You may stage and tokenize your data in previous cells without penalty.
Throughput (Tokens/Sec)	5 Points	The custom optimizer achieves high sample efficiency, processing a maximal number of tokens per second given the 16GB VRAM constraint.
Sub-Criterion	Point Value	Grading Standard
:---	:---	:---
WikiText-103 (Perplexity)	7.5 Points	Model achieves a perplexity score strictly < 20.0. Lower scores receive maximum points; scores \geq 20.0 receive 0 points.
HellaSwag (Commonsense)	7.5 Points	Model achieves an accuracy score strictly > 55.0\%. Higher scores receive maximum points; scores \leq 55.0\% receive 0 points.
PIQA (Physical Reasoning)	7.5 Points	Model achieves an accuracy score strictly > 65.0\%. Higher scores receive maximum points; scores \leq 65.0\% receive 0 points.
MT-Bench (Dialogue)	7.5 Points	Model achieves an instruction-following score strictly > 5.0. The model must output coherent, multi-turn conversational text without endless looping.
Sub-Criterion	Point Value	Grading Standard
:---	:---	:---
Mathematical Proofs	10 Points	The Writeup provides rigorous, unambiguous formal logic detailing how \Delta W is derived locally. Variables and functions are properly defined.
Code Readability	5 Points	The Kaggle Notebook and GitHub repository contain clean, PEP-8 compliant code with hyper-detailed docstrings explaining the custom tensor operations.
Strict Determinism	5 Points	Running the Kaggle Notebook yields the exact same benchmark scores on consecutive runs due to properly locked PRNG seeds across all libraries.
Phase 2: The Video Defense (Bonus Tie-Breaker)
In the event of a point tie between top submissions, the Judging Panel will use the attached 3-5 minute YouTube video to determine the final rankings.

Algorithmic Articulation: The primary author clearly and concisely explains the mathematical mechanism replacing BPTT on a digital whiteboard or architectural diagram.
Live Code Walkthrough: The team successfully points to the exact lines of raw tensor operations in the Kaggle Notebook that execute the localized weight updates.
Scientific Transparency: The team openly discusses the limitations, challenges, or bottlenecks discovered in their novel architecture during the 3-hour training window.
Judges
SohamThePal
Citation
SohamThePal. The Post-Backprop Challenge: Zero-Gradient Learning for Efficiency. https://kaggle.com/competitions/the-post-backprop-challenge-zero-gradient-learning-for-efficiency, 2026. Kaggle.
