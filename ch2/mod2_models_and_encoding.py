{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Lakshmi-Adhikari-AI/LLM-HuggingFace/blob/main/ch2/models_and_encoding.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AkK4hqyLeZhS"
      },
      "source": [
        "# Models (PyTorch)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pPUPXKBAeZhV"
      },
      "source": [
        "Install the Transformers, Datasets, and Evaluate libraries to run this notebook."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mYAyN3MLeZhV"
      },
      "outputs": [],
      "source": [
        "!pip install datasets evaluate transformers[sentencepiece]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zIWv1b7TeZhW"
      },
      "outputs": [],
      "source": [
        "from transformers import BertConfig, BertModel\n",
        "\n",
        "# Building the config\n",
        "config = BertConfig()\n",
        "\n",
        "# Building the model from the config\n",
        "model = BertModel(config)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9c6jJO9UeZhX",
        "outputId": "a71c5e3e-2d43-46a3-8af7-1af5d85bc016"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "BertConfig {\n",
              "  [...]\n",
              "  \"hidden_size\": 768,\n",
              "  \"intermediate_size\": 3072,\n",
              "  \"max_position_embeddings\": 512,\n",
              "  \"num_attention_heads\": 12,\n",
              "  \"num_hidden_layers\": 12,\n",
              "  [...]\n",
              "}"
            ]
          },
          "execution_count": null,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "print(config)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dtSgzRnSeZhX"
      },
      "outputs": [],
      "source": [
        "from transformers import BertConfig, BertModel\n",
        "\n",
        "config = BertConfig()\n",
        "model = BertModel(config)\n",
        "\n",
        "# Model is randomly initialized!"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 37,
      "metadata": {
        "id": "8X7h4pmieZhY"
      },
      "outputs": [],
      "source": [
        "from transformers import BertModel\n",
        "\n",
        "model = BertModel.from_pretrained(\"bert-base-cased\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "metadata": {
        "id": "mucZ3B76eZhY"
      },
      "outputs": [],
      "source": [
        "model.save_pretrained(\"directory_on_my_computer\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 39,
      "metadata": {
        "id": "wIcMC-BkeZhY"
      },
      "outputs": [],
      "source": [
        "sequences = [\"Hello!\", \"Cool.\", \"Nice!\"]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 38,
      "metadata": {
        "id": "R5FY8J3oeZhY"
      },
      "outputs": [],
      "source": [
        "encoded_sequences = [\n",
        "    [101, 7592, 999, 102],\n",
        "    [101, 4658, 1012, 102],\n",
        "    [101, 3835, 999, 102],\n",
        "]"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ================== Loading the Model ===================\n",
        "from transformers import AutoTokenizer, AutoModel\n",
        "\n",
        "# Load the base model and tokenizer using pretrained checkpoint\n",
        "# AutoModel returns a base Transformer model class (no task head like classification on top)\n",
        "# Downloads config.json, pytorch_model.bin etc., from Hugging Face Hub\n",
        "model = AutoModel.from_pretrained(\"bert-base-cased\")\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"bert-base-cased\")\n",
        "\n",
        "# The model variable now holds:\n",
        "# - model architecture from config.json\n",
        "# - learned weights from pytorch_model.bin\n",
        "\n",
        "print(model)\n",
        "print(\"Model Type:\", type(model))  # e.g., <class 'transformers.models.bert.modeling_bert.BertModel'>\n",
        "\n",
        "\n",
        "# ================== Encoding Text ===================\n",
        "\n",
        "# Input sentences (2 different lengths)\n",
        "inputs = [\"Iam learning Transformers\", \"But its hard to learn very veryy verrrry vererr\"]\n",
        "\n",
        "# Tokenize input with:\n",
        "# - padding=True → pad to the longest sentence in batch\n",
        "# - truncation=True → truncate if any sentence exceeds model's max length (default 512)\n",
        "# - max_length=12 → manually set cutoff length\n",
        "# - return_tensors=\"pt\" → return PyTorch tensors\n",
        "encoded_input = tokenizer(\n",
        "    inputs,\n",
        "    padding=True,\n",
        "    truncation=True,\n",
        "    max_length=12,\n",
        "    return_tensors=\"pt\"\n",
        ")\n",
        "\n",
        "print(\"\\nEncoded Input:\")\n",
        "print(encoded_input)\n",
        "\n",
        "# View tokens before they are converted to IDs\n",
        "print(\"\\nTokenized Words (Subwords):\")\n",
        "print(tokenizer.tokenize(inputs[0]))\n",
        "print(tokenizer.tokenize(inputs[1]))\n",
        "\n",
        "# Decode input_ids (convert back to readable string)\n",
        "print(\"\\nDecoded Inputs:\")\n",
        "for sequence_ids in encoded_input['input_ids']:\n",
        "    print(tokenizer.decode(sequence_ids))\n",
        "\n",
        "# Show final input_ids tensor and its shape\n",
        "print(\"\\nInput IDs Tensor:\")\n",
        "print(encoded_input['input_ids'])\n",
        "print(\"Shape:\", encoded_input['input_ids'].shape)  # (batch_size, sequence_length)\n"
      ],
      "metadata": {
        "id": "LQVDndlY2DdI",
        "outputId": "6ef6d946-8c24-4ea3-9a14-d498c20546b5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "BertModel(\n",
            "  (embeddings): BertEmbeddings(\n",
            "    (word_embeddings): Embedding(28996, 768, padding_idx=0)\n",
            "    (position_embeddings): Embedding(512, 768)\n",
            "    (token_type_embeddings): Embedding(2, 768)\n",
            "    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
            "    (dropout): Dropout(p=0.1, inplace=False)\n",
            "  )\n",
            "  (encoder): BertEncoder(\n",
            "    (layer): ModuleList(\n",
            "      (0-11): 12 x BertLayer(\n",
            "        (attention): BertAttention(\n",
            "          (self): BertSdpaSelfAttention(\n",
            "            (query): Linear(in_features=768, out_features=768, bias=True)\n",
            "            (key): Linear(in_features=768, out_features=768, bias=True)\n",
            "            (value): Linear(in_features=768, out_features=768, bias=True)\n",
            "            (dropout): Dropout(p=0.1, inplace=False)\n",
            "          )\n",
            "          (output): BertSelfOutput(\n",
            "            (dense): Linear(in_features=768, out_features=768, bias=True)\n",
            "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
            "            (dropout): Dropout(p=0.1, inplace=False)\n",
            "          )\n",
            "        )\n",
            "        (intermediate): BertIntermediate(\n",
            "          (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
            "          (intermediate_act_fn): GELUActivation()\n",
            "        )\n",
            "        (output): BertOutput(\n",
            "          (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
            "          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
            "          (dropout): Dropout(p=0.1, inplace=False)\n",
            "        )\n",
            "      )\n",
            "    )\n",
            "  )\n",
            "  (pooler): BertPooler(\n",
            "    (dense): Linear(in_features=768, out_features=768, bias=True)\n",
            "    (activation): Tanh()\n",
            "  )\n",
            ")\n",
            "Model Type: <class 'transformers.models.bert.modeling_bert.BertModel'>\n",
            "\n",
            "Encoded Input:\n",
            "{'input_ids': tensor([[  101,   146,  2312,  3776, 25267,   102,     0,     0,     0,     0,\n",
            "             0,     0],\n",
            "        [  101,  1252,  1157,  1662,  1106,  3858,  1304,  1304,  1183,  1396,\n",
            "         11096,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n",
            "        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],\n",
            "        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}\n",
            "\n",
            "Tokenized Words (Subwords):\n",
            "['I', '##am', 'learning', 'Transformers']\n",
            "['But', 'its', 'hard', 'to', 'learn', 'very', 'very', '##y', 've', '##rr', '##rry', 've', '##rer', '##r']\n",
            "\n",
            "Decoded Inputs:\n",
            "[CLS] Iam learning Transformers [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]\n",
            "[CLS] But its hard to learn very veryy verr [SEP]\n",
            "\n",
            "Input IDs Tensor:\n",
            "tensor([[  101,   146,  2312,  3776, 25267,   102,     0,     0,     0,     0,\n",
            "             0,     0],\n",
            "        [  101,  1252,  1157,  1662,  1106,  3858,  1304,  1304,  1183,  1396,\n",
            "         11096,   102]])\n",
            "Shape: torch.Size([2, 12])\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "Models (PyTorch)",
      "provenance": [],
      "include_colab_link": true
    },
    "language_info": {
      "name": "python"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
