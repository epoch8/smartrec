```bash
model_name = "als_v3"
model_version = str(int(datetime.now().timestamp()))
model = RecommenderALS(
    model_name=model_name,
    model_version=model_version,
)
dataset = get_data()
model.train(dataset)
model.save_model(BASE_LOCAL_DIR)
model.save_model_triton(fs, BASE_S3_URL, num_to_keep=2)
```