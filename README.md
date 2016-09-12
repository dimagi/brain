# Brain

Brain is Dimagi's system for predicting and mining CommCareHQ data

## API

### POST /train/
Given information on the features and target column, this will trigger Brain to build a machine learning model for the data.

Request Body

```json
{
	"database": "commcarehq_reporting",  // Defaults to commcarehq_reporting
	"table": "config_report_aspace_1b26b1874f894d93aa2fd3dcb567fd4b_0b78ca74",
	"port": 5432,  // Defaults to 5432
	"host": "hqdb0.internal.commcarehq.org",
	"features": [
		{
			"name": "age",
			"column": "age_23s2a"  // Defaults to name
		},
		{
			"name": "weight",
			"column": "weight_232xwa"
		}
		...
	],
	"target": {
		"name": "ltfu",
		"column": "ltfu_x23sx"
	},
	"options": {
		"algorithm": "svm",  // Defaults to svm
		... // Other algorithm parameters
	}
}
```

Return Body SUCCESS

```json
{
	"status": "ok",
	"model_id": "7d398287-aa9b-4858-8d1c-7fb543ea9f3b",
	"progress_url": "/progress/7d398287-aa9b-4858-8d1c-7fb543ea9f3b"
}
```

Return Body FAILURE

```json
{
	"status": "error",
	"reason": "Reason for failure"
}
```

### GET /progress/<model_id>

This will return information progress information on the model.

Return Body

```json
{
	"status": "ready"  // One of ready | pending | error
	"reason": null  // Only applicable when status is "error"
}
```

###  POST /predict/<model_id>

Given features, this will return a prediction.

Request Body

```json
{
	"features":[
		{
			"name": "age",  // Must match name from model creation
			"value": 23
		}
		...
	]
}
```

Return Body

```json
{
	"status": "ok"
	"reason": null  // Only applicable if status is "error"
	"prediction": {
		"name": "ltfu",
		"value": 0.73
	}
}
```

### POST /build_stats/<model_id>

Given a model id, will trigger algorithm to find various stats on the data.

Return Body 

```json
{
	"status": "ok",
	"progress_url": "/progress/stats/7d398287-aa9b-4858-8d1c-7fb543ea9f3b"
}
```

### GET /stats/<model_id>

Given a model ID will return various stats on the dataset if stats are finished building.

```json
{
	"status": "ok",
	"stats": {
		...
	}
}
```


