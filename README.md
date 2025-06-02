# capstone_2024
This project is submitted by Team U30A from the Department of Artificial Intelligence, College of Software and Convergence, Hanyang University ERICA for the 2025 Spring Capstone FAIR.

## Requirements

This project is developed using Python 3.10. Please upgrade the basic packages compatible with Python 3.10 before installing the requirements.txt.
```python
pip install --upgrade pip setuptools wheel
```
~~~python
pip install -r requirements.txt
~~~

Also, run the following command to use PyTorch with CUDA:
~~~python
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
~~~


## Table of Contents
- [Background](#Background)
- [Goal](#Goal)
- [Data_Proprocesing](#Data_Proprocesing)
- [Trajectory Classification](#Trajectory_Classification)
- [User Evaluation Metrics](#User_Evaluation_Metrics)
- [Trajectory Generation](#Trajectory_Generation)
----

### Background
South Korea's aging index continues to rise, and according to Statistics Korea data, the aging index has gradually increased from 2020 to 2025. This trend has drawn our attention to stroke, a disease that primarily affects the elderly population.

Stroke causes nerve damage that restricts upper limb movement, particularly shoulder mobility, making upper limb rehabilitation exercises essential for recovery. However, current shoulder rehabilitation programs operate with standardized approaches that fail to adequately consider individual patients' exercise performance capabilities.

Therefore, **this project aims to develop an AI-based personalized rehabilitation system to effectively support upper limb functional recovery for stroke patients.**

----

### Goal
1. **Rehabilitation Exercise Trajectory Classification and Recognition**: Understand the continuity of movement patterns and develop algorithms that consider joint-specific exercise characteristics.
2. **Exercise Performance Evaluation System**: Evaluate the accuracy and completeness of user trajectories and measure current status.
3. **Adaptive Exercise Trajectory Generation with Difficulty Adjustment**: Generate personalized exercise trajectories based on evaluation results and develop difficulty adjustment algorithms that consider performance capabilities.
----

### Data Preprocessing

We utilized trajectory data collected directly from Hexar Human Care's U30A device. After collecting 21 feature values and converting them into a structured format, we decomposed the robot's end-effector positions and joint angles into individual features, primarily working with 7 key data points. All data were unified into numerical format, and relative sequence values were converted to absolute time to ensure proper chronological ordering of the time-series data.
| **Circle** | **Arc** | **Line** |
|---|---|---|
| clock_b, clock_m, clock_t, clock_big, clock_l, clock_r, counter_b, counter_m, counter_t, counter_big, counter_l, counter_r | h_d, h_u, v_45, v_90, v_135, v_180 | d_r, d_l |

A total of 20 predefined trajectory types were collected. Data was gathered from 16 participants (9 women and 7 men), with each participant performing 100 repetitions per arm, resulting in 2,000 trajectory samples.

----

### Trajectory Classification
<img width="632" alt="image" src="https://github.com/user-attachments/assets/08fd2ce3-5af5-4d14-acf9-e8942c6b9cf6" />

> **Fig 1: Transformer Model Structure**

We designed a Transformer model to classify trajectory types from user input trajectories. The model uses only the Transformer encoder to process time-series data and output a single class label, directly connecting all time points in the trajectory data to assess the complete motion sequence.
The model leverages Multi-Head Attention to learn distinctive patterns and dependencies specific to each trajectory type, enabling recognition and classification based on the overall morphological characteristics of trajectories. Ultimately, it aggregates information from all time points into a unified vector representation to generate the trajectory classification label.

----

### User Evaluation Metrics
To assess the user's performance capability and generate appropriate rehabilitation exercise trajectories, evaluation metrics were developed for three types of trajectories: line, arc, and circle. To quantitatively analyze how users performed the trajectories, evaluation was conducted considering the geometric characteristics of the trajectories, focusing on whether the trajectories maintained close proximity to a plane when orthogonally projected in three-dimensional space, with additional evaluation of the unique characteristics of each trajectory type.

* ****Formula Description:****

**$$S_{corr} = 100 - \left(\frac{1}{n} \sum_{i=1}^{n} |S_{ori}| \times 100\right)$$**

- $S_{corr}$: Final accuracy score (0~100 points)
- $n$: Total number of evaluation metrics
- $S_{ori}$: Absolute difference between reference and user values
- $X_i$: i-th user measured value
- $S_i$: i-th reference standard value

> **Note:** $S_{ori} = X_i - S_i$, $X_i$ = i-th user value, $S_i$ = i-th reference value

* **Grading System**

| Grade | Score Range | 
|------|-----------|
| **Grade 1** | 76~100 points |
| **Grade 2** | 51~75 points |
| **Grade 3** | 26~50 points |
| **Grade 4** | 0~25 points |

Using these evaluation results, we dynamically adjust the interpolation weights between the user trajectory and target trajectory. For users with lower performance capabilities, the generated trajectory maintains a shape closer to their current movement pattern, while for users with superior performance, the generated trajectory approaches the ideal target trajectory shape.

### Trajectory Generation
<img width="599" alt="image" src="https://github.com/user-attachments/assets/e728eb43-a97d-4561-88c0-195f4eefa854" />
> **Fig 2: E2E Transformer Model Structure**


### Interactive Tutorial
### Conclusion

----
