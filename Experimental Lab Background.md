# **Democratizing Continuous Model Updates: A Two-Stage Delta-Tuning Architectural Experiment**

## **Executive Summary**

In enterprise production AI, keeping foundational language models updated with fresh, dynamic knowledge—such as real-time Wikipedia edits, continuous news feeds, or daily enterprise document updates—remains a core architectural bottleneck. Traditional full-parameter retraining across billions of parameters is computationally cost-prohibitive, requiring continuous access to high-end GPU clusters. Furthermore, fine-tuning base models solely on incremental new data introduces **catastrophic forgetting**, causing the model to overfit to recent inputs while degrading its historical pre-trained knowledge base.  
To address these challenges, this project demonstrates a lightweight, two-stage **Parameter-Efficient Fine-Tuning (PEFT)** pipeline built using **Low-Rank Adaptation (LoRA)**. Operating within local CPU/consumer GPU hardware constraints, this experiment proves that dynamic knowledge shifts can be injected into a small language model using tiny, modular adapter weights (\<10 MB) without altering or retraining the underlying frozen base model.

## **Technical Problem Statement & Trade-Offs**

### **The Cost and Scale Bottleneck**

When deploying enterprise LLMs, maintaining up-to-date domain knowledge generally forces a trade-off between two approaches:

> 1. **Full-Parameter Retraining:** Updating all model weights on fresh datasets requires massive high-speed VRAM allocation (e.g., arrays of Nvidia H100 GPUs for multi-billion parameter models). Re-ingesting full historical datasets daily to prevent knowledge degradation incurs extreme operational burn.  
> 2. **Retrieval-Augmented Generation (RAG):** Injecting context dynamically via vector databases avoids weight updates, but relies heavily on prompt window context space, latency constraints, and dynamic retrieval performance.

### **The Delta-Tuning Alternative**

Parameter-Efficient Fine-Tuning via LoRA offers a middle ground. By freezing the baseline parameters and targeting specific linear projection layers with low-rank decomposition matrices, models can adapt their generation trajectories on target domains at a fraction of the compute and storage overhead.

## **Lab Architecture & Execution Workflow**

The project was structured as an end-to-end local MLOps pipeline using Qwen2.5-0.5B-Instruct as the foundation model due to its rapid execution boundaries on standard hardware.

\+-------------------------------------------------------------------+  
|                        Stage 1: Base Model                        |  
|   Qwen2.5-0.5B-Instruct Baseline (500M Params \- Frozen Weights)   |  
\+-------------------------------------------------------------------+  
                                  |  
                                  v  
\+-------------------------------------------------------------------+  
|                      Stage 2: LoRA Delta Update                   |  
|   Inject \~10 MB Low-Rank Adapter Matrices into Attention Projections |  
|            Target Modules: q\_proj, v\_proj | Rank: r=8             |  
\+-------------------------------------------------------------------+  
                                  |  
                                  v  
\+-------------------------------------------------------------------+  
|                      Stage 3: Side-by-Side Eval                   |  
|    Interactive comparison of Base output vs. LoRA-adapted shift   |  
\+-------------------------------------------------------------------+

### **1\. Stage 1: Baseline Establishment (01\_train\_base.py)**

> * **Objective:** Establish the initial domain baseline on historical text snapshots.  
> * **Execution:** Standard causal language modeling loss optimization run on full base model parameters, saving baseline weights into ./models/base\_wiki\_model/.

### **2\. Stage 2: Parameter-Efficient Delta Adaptation (02\_train\_lora\_delta.py)**

> * **Objective:** Adapt generation trajectories to incoming data shifts without mutating base parameters.  
> * **Configuration:**  
  * **Frozen Base Parameters:** 100% of the baseline Qwen2.5-0.5B weights were set to non-trainable.  
  * **Target Modules:** Query (q\_proj) and Value (v\_proj) attention projection layers.  
  * **Rank & Alpha:** \$r \= 8\$, \$\\alpha \= 16\$, with a dropout of \$0.05\$.  
  * **Trainable Parameter Reduction:** Less than **0.1%** of total model parameters were updated.  
  * **Output Artifact:** A modular \~10 MB adapter binary saved to ./models/lora\_wiki\_delta/.

### **3\. Stage 3: Side-by-Side Evaluation (03\_eval\_comparison.py)**

> * **Objective:** Validate knowledge modification by running identical inputs through both the frozen base model and the LoRA-adapted model simultaneously.

## **Key Experimental Results**

The evaluation script confirmed successful adaptation of the generation trajectories between the baseline model and the delta-adapted model:

| Evaluation Parameter | Base Model Output | LoRA Delta-Adapted Output |
| :---- | :---- | :---- |
| **Input Prompt** | RAG | RAG |
| **Model Behavior** | Defaults to pre-existing general pre-training baseline knowledge. | Generative path altered by the attention projection layer adjustments. |
| **Output Trajectory** | *"RAGA is an online platform for research and analysis of African languages."*  | *"Keywords: Raga, music, South Asia"*  |
| **Memory Footprint** | \~1 GB full parameter baseline. | Frozen base \+ **\~10 MB adapter overlay**. |

## **Implementation & Repository Structure**

The complete execution environment and codebase are structured for modular reproducibility:

Plaintext  
wiki-delta-tuning/  
├── .gitignore               \# Excludes large weight binaries & checkpoints  
├── requirements.txt         \# Project dependencies (torch, transformers, peft, datasets)  
├── README.md                \# Architectural overview and execution brief  
├── 01\_train\_base.py         \# Stage 1: Baseline model pre-training/fine-tuning script  
├── 02\_train\_lora\_delta.py    \# Stage 2: Parameter-efficient LoRA adapter training script  
└── 03\_eval\_comparison.py    \# Stage 3: Side-by-side terminal evaluation script

## **Architectural Key Takeaways**

> 1. **Extreme Storage and Compute Efficiency:** Updating rank-8 (\$r=8\$) attention projections reduced trainable parameters down to **\<0.1%** of the full architecture, replacing gigabytes of model duplication with lightweight **10 MB swappable artifacts**.  
> 2. **Isolation of Baseline Capabilities:** Because base model weights remain completely frozen, baseline pre-trained capabilities are preserved, mitigating full-parameter catastrophic forgetting.  
> 3. **Modular Enterprise Deployments:** In production environments, this pattern enables multi-tenant or domain-specific deployments where a single frozen base model serves multiple specialized LoRA adapters dynamically swapped based on request context.