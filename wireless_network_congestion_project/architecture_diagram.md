# Wireless Network Congestion Predictor Architecture

```mermaid
flowchart LR
    classDef user fill:#f7f7f7,stroke:#555,stroke-width:1px,color:#111;
    classDef app fill:#e8f1fb,stroke:#2f6fad,stroke-width:1px,color:#111;
    classDef process fill:#eaf7ea,stroke:#2e8b57,stroke-width:1px,color:#111;
    classDef store fill:#fff4db,stroke:#b7791f,stroke-width:1px,color:#111;

    U[User / Analyst]:::user

    subgraph CH[Channel]
        F[Web Dashboard]:::app
    end

    subgraph API[Application Layer]
        B[FastAPI Backend<br/>main.py]:::app
        S1[Summary API]:::app
        S2[Training API]:::app
        S3[Prediction API]:::app
    end

    subgraph ML[Data and ML Layer]
        D[(IoT Dataset CSV)]:::store
        P[Preprocessing<br/>clean + features + EDA]:::process
        T[Model Training<br/>LR + RF + DT]:::process
    end

    subgraph ST[Saved Outputs]
        M[(Best Model)]:::store
        R[(Metrics)]:::store
        G[(Plots)]:::store
    end

    U --> F
    F -->|View dashboard| S1
    F -->|Run pipeline| S2
    F -->|Predict congestion| S3

    S1 --> B
    S2 --> B
    S3 --> B

    B --> D
    D --> P
    P --> T
    P --> G
    T --> M
    T --> R

    M --> S3
    R --> S1
    G --> S1
    S1 --> F
    S3 --> F
```

## Neat flow

- User interacts with the web dashboard.
- Dashboard calls the FastAPI APIs for summary, training, and prediction.
- Backend reads the IoT dataset and runs preprocessing plus model training.
- Best model, metrics, and plots are saved as outputs.
- Summary and prediction results are sent back to the dashboard.
