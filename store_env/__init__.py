from gymnasium import register

register(id="StoreEnv-v1", entry_point="store_env.environment.env:make_env")