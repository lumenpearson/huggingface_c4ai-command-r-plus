<p align="center">
  <a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus"><img alt="Header" 
src="./assets/header.png" /></a>
</p>

# Model Card for C4AI Command R+

🚨 **This model is non-quantized version of C4AI Command R+. You can find the quantized version of C4AI Command R+ using bitsandbytes [here](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit)**.


## Model Summary

C4AI Command R+ is an open weights research release of a 104B billion parameter model with highly advanced capabilities, this includes Retrieval Augmented Generation (RAG) and tool use to automate sophisticated tasks. The tool use in this model generation enables multi-step tool use which allows the model to combine multiple tools over multiple steps to accomplish difficult tasks. C4AI Command R+ is a multilingual model evaluated in 10 languages for performance: English, French, Spanish, Italian, German, Brazilian Portuguese, Japanese, Korean, Arabic, and Simplified Chinese. Command R+ is optimized for a variety of use cases including reasoning, summarization, and question answering.

C4AI Command R+ is part of a family of open weight releases from Cohere For AI and Cohere. Our smaller companion model is [C4AI Command R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)

Developed by: [Cohere](https://cohere.com/) and [Cohere For AI](https://cohere.for.ai)

- Point of Contact: Cohere For AI: [cohere.for.ai](https://cohere.for.ai/)
- License: [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license), requires also adhering to [C4AI's Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)
- Model: c4ai-command-r-plus
- Model Size: 104 billion parameters
- Context length: 128K

**Try C4AI Command R+**

You can try out C4AI Command R+ before downloading the weights in our hosted [Hugging Face Space](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus).

**Usage**

Please install `transformers` from the source repository that includes the necessary changes for this model.
```python
# pip install 'git+https://github.com/huggingface/transformers.git'
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-plus"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Format message with the command-r-plus chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

**Quantized model through bitsandbytes, 8-bit precision**

```python
# pip install 'git+https://github.com/huggingface/transformers.git' bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "CohereForAI/c4ai-command-r-plus"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# Format message with the command-r-plus chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

**Quantized model through bitsandbytes, 4-bit precision**

This model is non-quantized version of C4AI Command R+. You can find the quantized version of C4AI Command R+ using bitsandbytes [here](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit).

## Model Details

**Input**: Models input text only.

**Output**: Models generate text only.

**Model Architecture**: This is an auto-regressive language model that uses an optimized transformer architecture. After pretraining, this model uses supervised fine-tuning (SFT) and preference training to align model behavior to human preferences for helpfulness and safety.

**Languages covered**: The model is optimized to perform well in the following languages: English, French, Spanish, Italian, German, Brazilian Portuguese, Japanese, Korean, Simplified Chinese, and Arabic. 

Pre-training data additionally included the following 13 languages: Russian, Polish, Turkish, Vietnamese, Dutch, Czech, Indonesian, Ukrainian, Romanian, Greek, Hindi, Hebrew, Persian.

**Context length**: Command R+ supports a context length of 128K.

## Evaluations

Command R+ has been submitted to the [Open LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). We include the results below, along with a direct comparison to the strongest state-of-art open weights models currently available on Hugging Face. We note that these results are only useful to compare when evaluations are implemented for all models in a [standardized way](https://github.com/EleutherAI/lm-evaluation-harness) using publically available code, and hence shouldn't be used for comparison outside of models submitted to the leaderboard or compared to self-reported numbers which can't be replicated in the same way. 

| Model                           |   Average |   Arc (Challenge) |   Hella Swag |   MMLU |   Truthful QA |   Winogrande |   GSM8k |
|:--------------------------------|----------:|------------------:|-------------:|-------:|--------------:|-------------:|--------:|
| **CohereForAI/c4ai-command-r-plus** |      74.6 |             70.99 |         88.6 |   75.7 |          56.3 |         85.4 |    70.7 |
| [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)        |      74.5 |             68.9  |         89   |   73.7 |          66.9 |         81.8 |    66.9 |
| [Mixtral 8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)           |      72.7 |             70.1  |         87.6 |   71.4 |          65   |         81.1 |    61.1 |
| [Mixtral 8x7B Chat](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)               |      72.6 |             70.2  |         87.6 |   71.2 |          64.6 |         81.4 |    60.7 |
| [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)  |      68.5 |             65.5  |         87   |   68.2 |          52.3 |         81.5 |    56.6 |
| [Llama 2 70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)                     |      67.9 |             67.3  |         87.3 |   69.8 |          44.9 |         83.7 |    54.1 |
| [Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat)                     |      65.3 |             65.4  |         84.2 |   74.9 |          55.4 |         80.1 |    31.9 |
| [Gemma-7B](https://huggingface.co/google/gemma-7b)                       |      63.8 |             61.1  |         82.2 |   64.6 |          44.8 |         79   |    50.9 |
| [LLama 2 70B Chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)             |      62.4 |             64.6  |         85.9 |   63.9 |          52.8 |         80.5 |    26.7 |
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)                 |      61   |             60    |         83.3 |   64.2 |          42.2 |         78.4 |    37.8 |

We include these metrics here because they are frequently requested, but note that these metrics do not capture RAG, multilingual, tooling performance or the evaluation of open ended generations which we believe Command R+ to be state-of-art at. For evaluations of RAG, multilingual and tooling read more [here](https://txt.cohere.com/command-r-plus-microsoft-azure/). For evaluation of open ended generation, Command R+ is currently being evaluated on the [chatbot arena](https://chat.lmsys.org/). 

### Tool use & multihop capabilities:

Command R+ has been specifically trained with conversational tool use capabilities. These have been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template will likely reduce performance, but we encourage experimentation.

Command R+’s tool use functionality takes a conversation as input (with an optional user-system preamble), along with a list of available tools. The model will then generate a json-formatted list of actions to execute on a subset of those tools. Command R+ may use one of its supplied tools more than once.

The model has been trained to recognise a special `directly_answer` tool, which it uses to indicate that it doesn’t want to use any of its other tools. The ability to abstain from calling a specific tool can be useful in a range of situations, such as greeting a user, or asking clarifying questions.
We recommend including the `directly_answer` tool, but it can be removed or renamed if required.

Comprehensive documentation for working with command R+'s tool use prompt template can be found [here](https://docs.cohere.com/docs/prompting-command-r).

The code snippet below shows a minimal working example on how to render a prompt.

<details>
<summary><b>Usage: Rendering Tool Use Prompts [CLICK TO EXPAND]</b> </summary>

```python
from transformers import AutoTokenizer

model_id = "CohereForAI/c4ai-command-r-plus"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# define conversation input:
conversation = [
    {"role": "user", "content": "Whats the biggest penguin in the world?"}
]
# Define tools available for the model to use:
tools = [
  {
    "name": "internet_search",
    "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
    "parameter_definitions": {
      "query": {
        "description": "Query to search the internet with",
        "type": 'str',
        "required": True
      }
    }
  },
  {
    'name': "directly_answer",
    "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
    'parameter_definitions': {}
  }
]

# render the tool use prompt as a string:
tool_use_prompt = tokenizer.apply_tool_use_template(
    conversation,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
)
print(tool_use_prompt)
```

</details>

<details>
<summary><b>Example Rendered Tool Use Prompt [CLICK TO EXPAND]</b></summary>

````
<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

## Available Tools
Here is a list of tools that you have available to you:

```python
def internet_search(query: str) -> List[Dict]:
    """Returns a list of relevant document snippets for a textual query retrieved from the internet

    Args:
        query (str): Query to search the internet with
    """
    pass
```

```python
def directly_answer() -> List[Dict]:
    """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
    """
    pass
```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
```json
[
    {
        "tool_name": title of the tool in the specification,
        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
    }
]```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

````

</details>

<details>
<summary><b>Example Rendered Tool Use Completion [CLICK TO EXPAND]</b></summary>

````
Action: ```json
[
      {
          "tool_name": "internet_search",
          "parameters": {
              "query": "biggest penguin in the world"
          }
      }
]
```
````
</details>

### Grounded Generation and RAG Capabilities: 

Command R+ has been specifically trained with grounded generation capabilities. This means that it can generate responses based on a list of supplied document snippets, and it will include grounding spans (citations) in its response indicating the source of the information. This can be used to enable behaviors such as grounded summarization and the final step of Retrieval Augmented Generation (RAG). This behavior has been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template may reduce performance, but we encourage experimentation.

Command R+’s grounded generation behavior takes a conversation as input (with an optional user-supplied system preamble, indicating task, context and desired output style), along with a list of retrieved document snippets. The document snippets should be chunks, rather than long documents, typically around 100-400 words per chunk. Document snippets consist of key-value pairs. The keys should be short descriptive strings, the values can be text or semi-structured.

By default, Command R+ will generate grounded responses by first predicting which documents are relevant, then predicting which ones it will cite, then generating an answer. Finally, it will then insert grounding spans into the answer. See below for an example. This is referred to as `accurate` grounded generation.

The model is trained with a number of other answering modes, which can be selected by prompt changes. A `fast` citation mode is supported in the tokenizer, which will directly generate an answer with grounding spans in it, without first writing the answer out in full. This sacrifices some grounding accuracy in favor of generating fewer tokens.

Comprehensive documentation for working with Command R+'s grounded generation prompt template can be found [here](https://docs.cohere.com/docs/prompting-command-r).

The code snippet below shows a minimal working example on how to render a prompt.

<details>
<summary> <b>Usage: Rendering Grounded Generation prompts [CLICK TO EXPAND]</b> </summary>

````python
from transformers import AutoTokenizer

model_id = "CohereForAI/c4ai-command-r-plus"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# define conversation input:
conversation = [
    {"role": "user", "content": "Whats the biggest penguin in the world?"}
]
# define documents to ground on:
documents = [
    { "title": "Tall penguins", "text": "Emperor penguins are the tallest growing up to 122 cm in height." }, 
    { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."}
]

# render the tool use prompt as a string:
grounded_generation_prompt = tokenizer.apply_grounded_generation_template(
    conversation,
    documents=documents,
    citation_mode="accurate", # or "fast"
    tokenize=False,
    add_generation_prompt=True,
)
print(grounded_generation_prompt)
````
</details>

<details>
<summary><b>Example Rendered Grounded Generation Prompt [CLICK TO EXPAND]</b></summary>
  
````<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results>
Document: 0
title: Tall penguins
text: Emperor penguins are the tallest growing up to 122 cm in height.

Document: 1
title: Penguin habitats
text: Emperor penguins only live in Antarctica.
</results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.
Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
Thirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
````

</details>

<details>
<summary><b>Example Rendered Grounded Generation Completion [CLICK TO EXPAND]</b></summary>

````
Relevant Documents: 0,1
Cited Documents: 0,1
Answer: The Emperor Penguin is the tallest or biggest penguin in the world. It is a bird that lives only in Antarctica and grows to a height of around 122 centimetres.
Grounded answer: The <co: 0>Emperor Penguin</co: 0> is the <co: 0>tallest</co: 0> or biggest penguin in the world. It is a bird that <co: 1>lives only in Antarctica</co: 1> and <co: 0>grows to a height of around 122 centimetres.</co: 0>
````
</details>

### Code Capabilities:
Command R+ has been optimized to interact with your code, by requesting code snippets, code explanations, or code rewrites. It might not perform well out-of-the-box for pure code completion. For better performance, we also recommend using a low temperature (and even greedy decoding) for code-generation related instructions.

### Model Card Contact
For errors or additional questions about details in this model card, contact [info@for.ai](mailto:info@for.ai).

### Terms of Use: 
We hope that the release of this model will make community-based research efforts more accessible, by releasing the weights of a highly performant 104 billion parameter model to researchers all over the world. This model is governed by a [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license) License with an acceptable use addendum, and also requires adhering to [C4AI's Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy).

### Try Chat:
You can try Command R+ chat in the playground [here](https://dashboard.cohere.com/playground/chat). You can also use it in our dedicated Hugging Face Space [here](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus).