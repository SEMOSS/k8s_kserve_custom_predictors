# Gliner Predictor

## Build Docker Image
Run this command from the project root directory to build the image:
```bash
# CPU
docker build -f models/gliner/Dockerfile --build-arg MODE=cpu -t gliner-predictor:cpu .
# GPU
docker build -f models/gliner/Dockerfile --build-arg MODE=gpu -t gliner-predictor:gpu .
```

- Defaults to CPU

## Run Docker Container

```bash
docker run -p 8080:8080 --name gliner-predictor-cpu gliner-predictor:cpu
```

## Python Packages
- `gliner=>0.2.16`

## Example Endpoint
- `http://localhost:8080/v2/models/gliner-multi-v2-1/infer`

## Example Payload
```json
{
  "inputs": [
    {
      "name": "text",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["The French Revolution was a period of political and societal change in France that began with the Estates General of 1789, and ended with the coup of 18 Brumaire in November 1799 and the formation of the French Consulate. Many of its ideas are considered fundamental principles of liberal democracy, while its values and institutions remain central to modern French political discourse."]
    },
    {
      "name": "labels",
      "shape": [3],
      "datatype": "BYTES",
      "data": ["country", "year", "event"]
    },
    {
      "name": "mask_entities",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["year"]
    }
  ]
}
```

## Example Response
```json
{
    "model_name": "gliner-multi-v2-1",
    "model_version": null,
    "id": "159440f9-043c-4822-9a7e-34698f369547",
    "parameters": null,
    "outputs": [
        {
            "name": "output",
            "shape": [
                1
            ],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "The French Revolution was a period of political and societal change in France that began with the Estates General of m_5T7J50, and ended with the coup of 18 Brumaire in m_A6C6H1 and the formation of the French Consulate. Many of its ideas are considered fundamental principles of liberal democracy, while its values and institutions remain central to modern French political discourse."
            ]
        },
        {
            "name": "raw_output",
            "shape": [
                1
            ],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "[{\"start\": 4, \"end\": 21, \"text\": \"French Revolution\", \"label\": \"event\", \"score\": 0.774919867515564}, {\"start\": 71, \"end\": 77, \"text\": \"France\", \"label\": \"country\", \"score\": 0.9852698445320129}, {\"start\": 117, \"end\": 121, \"text\": \"1789\", \"label\": \"year\", \"score\": 0.8479412198066711}, {\"start\": 150, \"end\": 161, \"text\": \"18 Brumaire\", \"label\": \"event\", \"score\": 0.7921046614646912}, {\"start\": 165, \"end\": 178, \"text\": \"November 1799\", \"label\": \"year\", \"score\": 0.7803375720977783}]"
            ]
        },
        {
            "name": "mask_values",
            "shape": [
                1
            ],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "{\"November 1799\": \"m_A6C6H1\", \"m_A6C6H1\": \"November 1799\", \"1789\": \"m_5T7J50\", \"m_5T7J50\": \"1789\"}"
            ]
        },
        {
            "name": "input",
            "shape": [
                1
            ],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "The French Revolution was a period of political and societal change in France that began with the Estates General of 1789, and ended with the coup of 18 Brumaire in November 1799 and the formation of the French Consulate. Many of its ideas are considered fundamental principles of liberal democracy, while its values and institutions remain central to modern French political discourse."
            ]
        },
        {
            "name": "entities",
            "shape": [
                3
            ],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "country",
                "year",
                "event"
            ]
        }
    ]
}
```