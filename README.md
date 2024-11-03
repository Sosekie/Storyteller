# Storyteller

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