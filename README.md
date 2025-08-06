<meta name="robots" content="noindex">

# ZRing-sketch

ZRing is a lightweight sketching algorithm designed to estimate the weighted cardinality of dynamic data streams that include both insertions and deletions. It is particularly suited for large-scale streaming scenarios such as database analytics, blockchain log summarization, and deduplication tasks. This project provides a implementation of ZRing, including code for generating training data that can be used to support machine learning-based estimation. 

---

## 🛠️ Build Instructions

This project uses a standard Makefile for building. Make sure your environment supports C++17. 

```bash
# Clone the repository
git clone repo
cd ZRing-sketch

# Build the project
make
````

---

## 🚀 Running the Code

```bash
./main [max_processes] [repeat_count]
```

* `m`: (Optional) Method to estimate the weighted cardinality, including ZRingDME, QSketch, QSketchDME (default: ZRingDME)
* `r`: (Optional) Number of experiments to run (default: 10)
* `k`: (Optional) Allocated memory size in KB (default: 1)

**Example:**

```bash
./main -m ZRingDME
```

---

## ⚙️ Parameter Settings (in `main.cpp`)

**Input path:**

Set your dataset path here (each line in the file is a float number):

```cpp
string dataset_path = "./data.txt";
```

**Output path:**

```cpp
string output_file = "./ZRingset/test.npy";
```

---

## 📊 Output Format

The output `.npy` file is a 2D array where each row represents one run. The columns contain:

| Index        | Meaning                         |
| ------------ | ------------------------------- |
| `0` to `a-1` | Proportions of zeros per column |
| `a`          | $\log_2$(sketch size)        |
| `a+1`        | $\log_2$($Z$ size)                        |
| `a+2`        | True weighted cardinality       |
| `a+3`        | Estimated cardinality           |
| `a+4`        | Update time (seconds)           |
| `a+5`        | Estimation time (seconds)       |

To test the method `ZRing-MLP`, run:
```bash
python test.py
```

---

## 📈 Example Output

After execution, you will see results like:

```
 AARE across all files: 0.211
 Time across all files: 0.014s
 EstiTime across all files: 0.003s
All files processed. Combined results saved to ./Trainset/test.npy
```

---

## 🧩 Input Format

The input dataset must be a `.txt` file where each line contains a float value (the weight of an element):

```
1.0
2.3
0.7
...
```

