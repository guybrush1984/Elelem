"""
Model definitions with capabilities and pricing for all supported providers
"""

MODELS = {
    # OpenAI Models (Standard tier pricing)
    "openai:gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 2.00,  # $2.00 per 1M input tokens
            "output_cost_per_1m": 8.00,  # $8.00 per 1M output tokens
            "currency": "USD"
        }
    },
    "openai:gpt-4.1-mini": {
        "provider": "openai",
        "model_id": "gpt-4.1-mini",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 0.40,  # $0.40 per 1M input tokens
            "output_cost_per_1m": 1.60,  # $1.60 per 1M output tokens
            "currency": "USD"
        }
    },
    "openai:gpt-5": {
        "provider": "openai",
        "model_id": "gpt-5",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 1.25,
            "output_cost_per_1m": 10.00,
            "currency": "USD"
        }
    },
    "openai:gpt-5-mini": {
        "provider": "openai",
        "model_id": "gpt-5-mini",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 0.25,
            "output_cost_per_1m": 2.00,
            "currency": "USD"
        }
    },
    "openai:o3": {
        "provider": "openai",
        "model_id": "o3",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": False  # o-series models don't support temperature
        },
        "cost": {
            "input_cost_per_1m": 2.00,
            "output_cost_per_1m": 8.00,
            "currency": "USD"
        }
    },
    "openai:o3-mini": {
        "provider": "openai",
        "model_id": "o3-mini",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": False,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 1.10,
            "output_cost_per_1m": 4.40,
            "currency": "USD"
        }
    },
    
    # GROQ Models
    "groq:openai/gpt-oss-120b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",  # GROQ's model ID format
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 0.15,
            "output_cost_per_1m": 0.75,
            "currency": "USD"
        }
    },
    "groq:openai/gpt-oss-20b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 0.10,
            "output_cost_per_1m": 0.50,
            "currency": "USD"
        }
    },
    "groq:moonshotai/kimi-k2-instruct": {
        "provider": "groq",
        "model_id": "moonshotai/kimi-k2-instruct",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 1.00,
            "output_cost_per_1m": 3.00,
            "currency": "USD"
        }
    },
    "groq:meta-llama/llama-4-maverick-17b-128e-instruct": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 0.20,
            "output_cost_per_1m": 0.60,
            "currency": "USD"
        }
    },
    "groq:meta-llama/llama-4-scout-17b-16e-instruct": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 0.11,
            "output_cost_per_1m": 0.34,
            "currency": "USD"
        }
    },
    
    # DeepInfra Models - CRITICAL: ALL supports_json_mode = False
    "deepinfra:openai/gpt-oss-120b": {
        "provider": "deepinfra",
        "model_id": "openai/gpt-oss-120b",
        "capabilities": {
            "supports_json_mode": False,  # DeepInfra doesn't support response_format
            "supports_temperature": True,
            "supports_system": True
        },
        "cost": {
            "input_cost_per_1m": 0.09,
            "output_cost_per_1m": 0.45,
            "currency": "USD"
        }
    },
    "deepinfra:openai/gpt-oss-20b": {
        "provider": "deepinfra",
        "model_id": "openai/gpt-oss-20b",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True
        },
        "cost": {
            "input_cost_per_1m": 0.04,
            "output_cost_per_1m": 0.16,
            "currency": "USD"
        }
    },
    "deepinfra:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "provider": "deepinfra",
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True
        },
        "cost": {
            "input_cost_per_1m": 0.08,  # Need to verify - using Scout pricing as placeholder
            "output_cost_per_1m": 0.30,
            "currency": "USD"
        }
    },
    "deepinfra:meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "provider": "deepinfra",
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True
        },
        "cost": {
            "input_cost_per_1m": 0.08,
            "output_cost_per_1m": 0.30,
            "currency": "USD"
        }
    },
    "deepinfra:moonshotai/Kimi-K2-Instruct": {
        "provider": "deepinfra",
        "model_id": "moonshotai/Kimi-K2-Instruct",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True
        },
        "cost": {
            "input_cost_per_1m": 0.50,
            "output_cost_per_1m": 2.00,
            "currency": "USD"
        }
    },
    "deepinfra:deepseek-ai/DeepSeek-R1-0528": {
        "provider": "deepinfra",
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True
        },
        "cost": {
            "input_cost_per_1m": 0.50,
            "output_cost_per_1m": 2.15,
            "currency": "USD"
        }
    }
}