```bash

# Create dataset with both user and item features
dataset = Dataset.construct(
    interactions_df=df_interactions,
)

# Create instance of the model
model_name = "als_build_ideas"
model_version = str(int(datetime.now().timestamp()))
model = RecommenderALS(
    model_name=model_name,
    model_version=model_version,
    recsys_config=recsys_config,
)

# Fit models
model.train(dataset)

# Save model to local and s3 dirs
model.save_model(BASE_LOCAL_DIR)
model.save_model_triton(base_s3_url=BASE_S3_URL, num_to_keep=2)
```