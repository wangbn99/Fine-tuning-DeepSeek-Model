# Fine-tuning DeepSeek Model with Financial Dataset

This project demonstrates how to fine-tune a DeepSeek language model for a financial question-answering task. The included Jupyter notebook (`fine-tuning-deepseek-model-with-financial-dataset.ipynb`) walks through the process from setting up the environment to training the model and preparing it for inference.

## Notebook Overview

The Jupyter notebook covers the following key steps:

1.  **Install Required Libraries:** Installs the necessary Python packages, primarily `transformers` and `datasets`.
2.  **Load the Model and Tokenizer:** Loads a pre-trained DeepSeek model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) and its corresponding tokenizer from the Hugging Face Hub.
3.  **Prepare the Input:** Defines a prompt structure for question answering and prepares an example.
4.  **Generate Text (Initial):** Demonstrates text generation with the base pre-trained model.
5.  **Load the Dataset:** Loads the `virattt/financial-qa-10K` dataset from Hugging Face, which contains question-answer pairs based on financial documents.
6.  **Preprocessing the Dataset:** Tokenizes the dataset to prepare it for training.
7.  **Train Model:** Sets up training arguments and fine-tunes the model on the financial Q&A dataset.
8.  **Model Inference after Fine-tuning:** (Cell not executed in the original notebook) Intended to show how to use the fine-tuned model to answer questions.
9.  **Save Model Locally:** (Cell not executed in the original notebook) Provides code to save the fine-tuned model to local storage.
10. **Push Model to HuggingFace Hub:** (Cell not executed in the original notebook) Provides code to upload the fine-tuned model to the Hugging Face Model Hub.

## Setup/Installation

To run the notebook, you'll need to install the following libraries:

```bash
pip install transformers datasets
```

You will also need a Python environment (e.g., Conda, venv) and Jupyter Notebook or JupyterLab. If you plan to train the model, a GPU with sufficient memory is highly recommended.

## Usage

1.  Ensure you have the required libraries installed (see Setup/Installation).
2.  Open the `fine-tuning-deepseek-model-with-financial-dataset.ipynb` notebook in Jupyter Notebook or JupyterLab.
3.  Run the cells sequentially to execute the steps.

## Dataset

The model is fine-tuned on the `virattt/financial-qa-10K` dataset, available on the Hugging Face Hub. This dataset is designed for question answering tasks in the financial domain.

## Model

The base model used for fine-tuning is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. This is a distilled version of the DeepSeek model, optimized for efficiency.

## Potential Issues and Considerations

*   **Out of Memory:** The training process (cell 8 in the notebook) encountered an `OutOfMemoryError` (`torch.OutOfMemoryError: CUDA out of memory`) with the provided setup (14.74 GiB GPU memory, batch size 16).
    *   **Possible Solutions:**
        *   Reduce `per_device_train_batch_size` in the `TrainingArguments`.
        *   Use gradient accumulation (`gradient_accumulation_steps` in `TrainingArguments`).
        *   Utilize a machine with more GPU memory.
        *   Consider techniques like LoRA (Low-Rank Adaptation) for more memory-efficient fine-tuning, although this would require modifying the training script.
*   **Execution of Later Cells:** Cells for inference with the fine-tuned model, saving the model, and pushing it to the Hugging Face Hub were not executed in the provided notebook. These will only work correctly after the model has been successfully trained.

## Future Work/Next Steps

*   Successfully train the model by addressing the memory issues.
*   Run the inference cells to test the performance of the fine-tuned model.
*   Save the fine-tuned model and tokenizer locally.
*   Optionally, push the fine-tuned model to the Hugging Face Model Hub for sharing and later use.
