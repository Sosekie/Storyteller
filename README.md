# Storyteller

## Project Requirements and Steps

### Project Objective
- [x] **Build a queueing model to predict the average latency of inference jobs for tabular/image diffusion systems.**
  - We developed a pipeline that integrates an LLM for story generation and an LDM for illustration generation. The pipeline includes configurable parameters for experimentation, which allows us to measure inference latency under various settings.
  
- [x] **Implement and evaluate early exit mechanisms to improve latency.**
  - We incorporated parallel and sequential generation strategies to explore different performance trade-offs, which act as latency optimization mechanisms.

---

### Steps to Achieve the Objective

#### **1. System Setup and Emulation**
- [x] **Set up or emulate a diffusion serving system capable of receiving and processing inference jobs.**
  - We implemented a pipeline that simulates a diffusion serving system. It processes a single prompt through an LLM to generate a story, then uses an LDM to create corresponding images.

---

#### **2. Mathematical Modeling**
- [ ] **Develop mathematical models to predict the average job response time of the serving system.**
  - We have not yet developed any explicit mathematical models to predict response times.

---

#### **3. Arrival Patterns**
- [ ] **Determine or assume the arrival patterns of inference jobs:**
  - We have not yet specified the average arrival rate or inter-arrival time distribution of inference jobs.

---

#### **4. Job Execution Time**
- [x] **Identify or estimate the execution time for inference jobs:**
  - We measured the execution time for both the LLM and LDM components under different system configurations.
  
- [x] **Conduct a systematic analysis of variance (ANOVA) to model execution times as functions of system parameters.**
  - We used DOE to explore the relationship between execution times and parameters like `llm_tokens` and `image_size`.

---

#### **5. Queueing System Assumptions**
- [ ] **Simplify and assume an underlying queueing system:**
  - We have not yet modeled the serving system as a queueing system (e.g., single queue and single server) or applied relevant queueing theory formulas.
  
- [ ] **Evaluate the accuracy of the queueing model:**
  - No comparison or error analysis between predicted and actual response times has been performed.

---

#### **6. Intermediate Project Milestone**
- [ ] **Finalize and validate the initial queueing model.**
  - This task is pending, as no queueing model has been developed yet.
  
- [ ] **Perform an evaluation of the latency predictions.**
  - Latency evaluations have been conducted at the experimental level but not in the context of a predictive queueing model.

---

#### **7. Optimization Strategies**
- [x] **Research and apply optimization techniques to improve system latency:**
  - We explored different generation strategies (`parallel` and `sequential`) to optimize LDM performance under varying workloads.
  
- [ ] **Verify the impact of the chosen optimization strategies.**
  - While strategies were implemented, their impact on latency has not been systematically verified using a model.

---

#### **8. System Performance Verification**
- [ ] **Evaluate whether the optimization strategies improve latency:**
  - Although experiments were conducted, the evaluation is not yet linked to a validated queueing model.
  
- [ ] **Ensure the queueing model captures the impact of the optimizations.**
  - No queueing model exists to assess the effectiveness of the optimizations.

---

### Final Deliverables
- [ ] **A detailed report explaining the project workflow and results.**
  - A comprehensive report covering queueing model development, statistical analysis, and optimizations has not yet been compiled.

- [x] **Supporting files such as code and simulation data.**
  - We saved experiment configurations, generated stories, images, and metrics as output files for further analysis.


what is arrival rate, how many queqe, check variability, check M/M/k or M/G/k, dynamic load balancing: join-shortest-queqe





## Environment

conda create --name storyteller python==3.10

conda activate storyteller

pip install -r requirements.txt

pip install -U diffusers

pip install --upgrade bitsandbytes

## Huggingface Token:

huggingface-cli login

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

## Lydia's feedback:

Wow you have proposed a very elaborate generation service. Lets try stay in more conceise scope and think about the following questions.  In terms of context of generation service: what are the key difference between the real time part and existing solution shown in your fig 1? I think the key differences are three inferences methods, which result into different coherence in quality and inference time.  In the phase one, you shall build a simple linear model to get the job execution times based on different inference methods and other factors. The second phase can be simply decide which inference method to use based on the "number" of oustanding jobs.  You actually don't need to try those methods mentioned in 3.6. Of course, you can but chose one direction and go deeper.  Try to setup basic sekleton of your system asap such that we can discuss the optimization aspects in the mideterm. 

## What to do next:

1. build a baseline based on sequential generation.
- LLM
- LDM
- pipeline

2. end-to-end parallel generation.
- change channel size of LDM based on number of prompts and corresponding images.

3. autoregressive generation(optional).
- do sequential generation but autogressively based on previous image and prompts.

4. model quantization.
- quantizing the model with diffusers(LDM I choose offer existing api for quantization)
- compare execution time before and after quantization, organize as table and save in latex.

5. do early exit.
- Use an evaluation metric to test when to exit early(FID, mutual imformation, etc).
- compare execution time before and after early exit, organize as table and save in latex.