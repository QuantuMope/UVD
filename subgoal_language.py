from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import time

# Initialize the LLM
# llm = LLM(
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     dtype="bfloat16",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.7
# )
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    dtype="bfloat16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=2048
)

# Configure generation parameters
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=256  # Adjust based on your needs
)

# # Load your prompts (assuming from a file)


# with open("prompts.json", "r") as f:
#     prompts = json.load(f)  # List containing your prompts

prompts = [
 '<｜User｜> How are you? <｜Assistant｜>'
]

# Process one prompt at a time
results = []
start_time = time.time()

for i, prompt in enumerate(tqdm(prompts)):
    # Generate response for a single prompt
    outputs = llm.generate([prompt], sampling_params)

    # Extract and store the generated text
    generated_text = outputs[0].outputs[0].text

    results.append({
        "prompt_id": i,
        "prompt": prompt,
        "generated_text": generated_text
    })

    # # Optionally save results periodically (e.g., every 100 prompts)
    # if (i + 1) % 100 == 0:
    #     with open(f"results_checkpoint_{i + 1}.json", "w") as f:
    #         json.dump(results, f, indent=2)

# Calculate and display total processing time
elapsed_time = time.time() - start_time
print(f"Processed {len(prompts)} prompts in {elapsed_time:.2f} seconds")
print(f"Average time per prompt: {elapsed_time / len(prompts):.4f} seconds")

print(results)

# # Save final results
# with open("results_final.json", "w") as f:
#     json.dump(results, f, indent=2)