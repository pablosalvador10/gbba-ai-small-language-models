from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
from typing import List, Dict

import os
import mlflow
from mlflow.models import infer_signature
import argparse
import sys
import logging

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from datetime import datetime

logger = logging.getLogger(__name__)

def load_model(model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
               use_cache: bool = False,
               trust_remote_code: bool = True,
               torch_dtype: torch.dtype = torch.bfloat16,
               device_map: dict = None,
               max_seq_length: int = 4096) -> tuple:
    """
    Loads a pre-trained model and its tokenizer with specified configurations.

    Parameters:
    - model_name_or_path (str): Identifier for the model to load. Can be a model ID or path.
    - use_cache (bool): Whether to use caching for model outputs.
    - trust_remote_code (bool): Whether to trust remote code when loading the model.
    - torch_dtype (torch.dtype): Data type for model tensors. Recommended to use torch.bfloat16 for efficiency.
    - device_map (dict): Custom device map for distributing the model's layers across devices.
    - max_seq_length (int): Maximum sequence length for the tokenizer.

    Returns:
    - tuple: A tuple containing the loaded model and tokenizer.
    """
    model_kwargs = {
        "use_cache": use_cache,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": device_map
    }
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = max_seq_length
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"
    return model, tokenizer

def convert_to_chat_format(df: pd.DataFrame) -> List[List[Dict[str, str]]]:
    """
    Converts a DataFrame containing questions and their types into a chat format.

    Parameters:
    - df (pd.DataFrame): A DataFrame with at least two columns: 'Question' and 'Kind of Query'.
      Each row represents a user query and its categorization.

    Returns:
    - List[List[Dict[str, str]]]: A list of chats, where each chat is a list of messages.
      Each message is a dictionary with 'role' and 'content' keys. The 'role' can be 'system',
      'user', or 'assistant', indicating the sender of the message. The 'content' is the text of the message.
    """
    chats = []

    for index, row in df.iterrows():
        chat = [
            {
                "role": "system",
                "content": "You are an AI assistant supporting users by categorizing their queries."
            },
            {
                "role": "user",
                "content": row["Question"]
            },
            {
                "role": "assistant",
                "content": f"This query is a '{row['Kind of Query']}' type."
            }
        ]
        chats.append(chat)
    
    return chats

def convert_chats_to_dataframe(chats: List[List[Dict[str, str]]]) -> pd.DataFrame:
    """
    Converts a list of chats into a DataFrame where each chat is represented as a dictionary in the 'message' column.

    Parameters:
    - chats (List[List[Dict[str, str]]]): A list of chats, where each chat is a list of messages.
      Each message is a dictionary with 'role' and 'content' keys.

    Returns:
    - pd.DataFrame: A DataFrame with a single column 'message', where each row contains a dictionary
      representing a chat.
    """
    # Convert each chat into a dictionary and store it in a list
    chat_dicts = [{'message': chat} for chat in chats]
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(chat_dicts)
    
    return df


def apply_chat_template(
    example: dict,
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """
    Applies a chat template to the messages in an example from a dataset.

    This function modifies the input example by adding a system message at the beginning
    if it does not already start with one. It then applies a chat template formatting
    using the specified tokenizer.

    Parameters:
    - example (dict): A dictionary representing a single example from a dataset. It must
      contain a key 'messages', which is a list of message dictionaries. Each message
      dictionary should have 'role' and 'content' keys.
    - tokenizer (PreTrainedTokenizer): An instance of a tokenizer that supports the
      `apply_chat_template` method for formatting chat messages.

    Returns:
    - dict: The modified example dictionary with an added 'text' key that contains the
      formatted chat as a string.
    """
    messages = example["message"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

def main(args):
    
    ###################
    # Hyper-parameters
    ###################
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key    
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    use_wandb = len(args.wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0) 

    training_config = {
        "bf16": True,
        "do_eval": False,
        "learning_rate": args.learning_rate,
        "log_level": "info",
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_train_epochs": args.epochs,
        "max_steps": -1,
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "remove_unused_columns": True,
        "save_steps": args.save_steps,
        "save_total_limit": 1,
        "seed": args.seed,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": args.grad_accum_steps,
        "warmup_ratio": args.warmup_ratio,
    }

    peft_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        #"target_modules": "all-linear",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": None,
    }

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    train_conf = TrainingArguments(
        **training_config,
        report_to="wandb" if use_wandb else "azure_ml",
        run_name=args.wandb_run_name if use_wandb else None,    
    )
    peft_conf = LoraConfig(**peft_config)
    model, tokenizer = load_model(args)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_conf}")    

    ##################
    # Data Processing
    ##################

    train_dataset = load_dataset('json', data_files=os.path.join(args.train_dir, 'train.jsonl'), split='train')
    eval_dataset = load_dataset('json', data_files=os.path.join(args.train_dir, 'eval.jsonl'), split='train')
    column_names = list(train_dataset.features)

    train_data = pd.read_csv(args.train_dir)
    train_data_chat_format = convert_to_chat_format(train_data)
    df_train_data_chat_format = convert_chats_to_dataframe(train_data_chat_format)
    train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(df_train_data_chat_format, columns=["message"]),split= "train")

    eval_data = pd.read_csv(args.eval_data)
    eval_data_chat_format = convert_to_chat_format(eval_data)
    df_eval_data_chat_format = convert_chats_to_dataframe(eval_data_chat_format)
    eval_data_dataset = datasets.Dataset.from_pandas(pd.DataFrame(df_eval_data_chat_format, columns=["message"]),split="train")

    column_names = list(train_dataset.features)

    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )

    with mlflow.start_run() as run:        
        ###########
        # Training
        ###########
        trainer = SFTTrainer(
            model=model,
            args=train_conf,
            peft_config=peft_conf,
            train_dataset=processed_train_dataset,
            eval_dataset=processed_eval_dataset,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            tokenizer=tokenizer,
            packing=True,
        )

        # Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
        
        last_checkpoint = None
        if os.path.isdir(checkpoint_dir):
            checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)]
            if len(checkpoints) > 0:
                checkpoints.sort(key=os.path.getmtime, reverse=True)
                last_checkpoint = checkpoints[0]        

        trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

        #############
        # Logging
        #############
        metrics = trainer_stats.metrics

        # Show final memory and time stats 
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

        logger.info(f"{metrics['train_runtime']} seconds used for training.")
        logger.info(f"{round(metrics['train_runtime']/60, 2)} minutes used for training.")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
                
        trainer.log_metrics("train", metrics)

        model_info = mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            #prompt_template=prompt_template,
            #signature=signature,
            artifact_path=args.model_dir,  # This is a relative path to save model files within MLflow run
        )


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()
    # curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # hyperparameters
    parser.add_argument("--train_dir", default="data", type=str, help="Input directory for training")
    parser.add_argument("--model_dir", default="./model", type=str, help="output directory for model")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--output_dir", default="./output_dir", type=str, help="directory to temporarily store when training a model")
    parser.add_argument("--train_batch_size", default=8, type=int, help="training - mini batch size for each gpu/process")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="evaluation - mini batch size for each gpu/process")
    parser.add_argument("--learning_rate", default=5e-06, type=float, help="learning rate")
    parser.add_argument("--logging_steps", default=2, type=int, help="logging steps")
    parser.add_argument("--save_steps", default=100, type=int, help="save steps")    
    parser.add_argument("--grad_accum_steps", default=4, type=int, help="gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--warmup_ratio", default=0.2, type=float, help="warmup ratio")
    parser.add_argument("--max_seq_length", default=2048, type=int, help="max seq length")
    parser.add_argument("--save_merged_model", type=bool, default=False)

    # lora hyperparameters
    parser.add_argument("--lora_r", default=16, type=int, help="lora r")
    parser.add_argument("--lora_alpha", default=16, type=int, help="lora alpha")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="lora dropout")
    
    # wandb params
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_watch", type=str, default="gradients") # options: false | gradients | all
    parser.add_argument("--wandb_log_model", type=str, default="false") # options: false | true

    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    #sys.argv = ['']
    args = parse_args()
    main(args)
