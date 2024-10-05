# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CALM training script for finetuning.

Hugging Face Trainer is used to train the CALM model.
Reference:
https://huggingface.co/docs/transformers/main_classes/trainer
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import datasets
from model import calm
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
import json

_ANCHOR_MODEL_DIR = flags.DEFINE_string(
    'anchor_model_dir', None, 'anchor model path.'
)
_AUG_MODEL_DIR = flags.DEFINE_string('aug_model_dir', None, 'aug model path.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'output directory.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 2e-5, 'learning rate.')
_EPOCHS = flags.DEFINE_integer('epochs', 100, 'number of epochs.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'batch size.')
_NUM_HEADS = flags.DEFINE_integer('num_heads', 1, 'number of heads.')
_NUM_CONNECTIONS = flags.DEFINE_integer(
    'num_connections', 2, 'number of connections.'
)
_CONNECTIONS = flags.DEFINE_list(
    'connections',
    None,
    'connections between the anchor and aug model. You cannot provide both'
    'connections and num_connections simultaneously.',
)
_EVAL_STEPS = flags.DEFINE_integer('eval_steps', 50, 'eval steps.')
_LOGGING_STEPS = flags.DEFINE_integer('logging_steps', 50, 'logging steps.')
_SAVE_STEPS = flags.DEFINE_integer('save_steps', 200, 'save steps.')
_MAX_STEPS = flags.DEFINE_integer('max_steps', 2000, 'max steps.')

import matplotlib.pyplot as plt
def plot_losses(trainer, output_dir):
    # 提取训练和评估损失
    logs = trainer.state.log_history
    train_loss = [log['loss'] for log in logs if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
    steps = range(len(train_loss))
    # every step * save_steps
    steps = [step * _LOGGING_STEPS.value for step in steps]
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss, label='Train Loss')
    plt.plot(steps, eval_loss, label='Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)

    # 保存为PNG文件
    plt.savefig(f'{output_dir}/loss_curve.png')
    plt.close()

def train(argv: Sequence[str]) -> None:
    del argv  # Unused.
    anchor_model_path = _ANCHOR_MODEL_DIR.value
    aug_model_path = _AUG_MODEL_DIR.value
    num_heads = _NUM_HEADS.value
    num_connections = _NUM_CONNECTIONS.value
    logging.info('anchor_model_path: %s', anchor_model_path)
    logging.info('aug_model_path: %s', aug_model_path)
    logging.info('Loading Tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(anchor_model_path)
    logging.info('Loading Composed Model...')

    calm_config = calm.CALMConfig(
        anchor_model=anchor_model_path,
        aug_model=aug_model_path,
        anchor_config=None,
        aug_config=None,
        num_connections=num_connections,
        num_heads=num_heads,
    )
    calm_config.save_pretrained("calm_saves/tmp/")
    print("calm_config saved to calm_saves/tmp/")
    calm_config = calm.CALMConfig.from_pretrained("calm_saves/tmp/")
    model = calm.CALM(calm_config)

    def load_dialogue_data(file_path):
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        qa_pairs = []
        for item in data:
            dialogue = item['dialogue']
            for i in range(0, len(dialogue), 2):
                if i + 1 < len(dialogue):
                    user_turn = dialogue[i]
                    assistant_turn = dialogue[i + 1]
                    if user_turn['role'] == 'user' and assistant_turn['role'] == 'assistant':
                        qa_pairs.append({
                            'instruction': user_turn['content'],
                            'input': '',
                            'output': assistant_turn['content']
                        })
        return qa_pairs

    def preprocess_function(examples):
        inputs = [ex for ex in examples['instruction']]
        targets = [ex for ex in examples['output']]
        model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=256)
        labels = tokenizer(targets, truncation=True, padding='max_length', max_length=256)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs


    # 加载对话数据并转换为QA格式
    qa_data = load_dialogue_data('filtered_familyclic_sc.json')
    full_data = datasets.Dataset.from_dict({'instruction': [d['instruction'] for d in qa_data],
                                             'input': [d['input'] for d in qa_data],
                                             'output': [d['output'] for d in qa_data]})
    # print(full_data)
    train_data, eval_data = full_data.train_test_split(test_size=0.1).values()
    train_data = train_data.map(preprocess_function, batched=True)
    eval_data = eval_data.map(preprocess_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    epochs = _EPOCHS.value
    batch_size = _BATCH_SIZE.value
    learning_rate = _LEARNING_RATE.value
    output_dir = _OUTPUT_DIR.value
    eval_steps = _EVAL_STEPS.value
    logging_steps = _LOGGING_STEPS.value
    save_steps = _SAVE_STEPS.value
    max_steps = _MAX_STEPS.value
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='steps',
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        label_names=[],
        report_to=['wandb'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.can_return_loss = True

    trainer.train()
    plot_losses(trainer, output_dir)
    trainer.save_model(
        output_dir,
    )

    print(f'Training complete! Model saved to {output_dir}')

if __name__ == '__main__':
    app.run(train)