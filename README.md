### Obama Digital Twin

This project introduces a ChatBot that captures the style of President Barack Obama. `Qwen2.5-7b` is aligned to capture Obama's style using Direct Policy Optimization.
The data for the alignment is based on [all obama speeches](https://github.com/q-n-t-m/obamas-speeches/tree/master). To reproduce the digital twin, do the following


1. Extract the speeches and question answers from [all obama speeches](https://github.com/q-n-t-m/obamas-speeches/tree/master)
```
python preprocess_speeches.py
```
2. Generate alignment dataset using Gemini by generate alternative speeches and answers for different presidents
```
python preprocess_speeches.py
```

3. Run direct policy optimization
```
python train_dop.py
```

