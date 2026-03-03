# STS (SAE-based Transferability Score)

This repository includes a PyTorch implementation of the ICLR 2026 paper [SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training](https://openreview.net/forum?id=KQYnfeBNjl) authored by Qi Zhang*, [Yifei Wang*](https://yifeiwang77.github.io/), Xiaohan Wang, Jiajun Chai, Guojun Yin, Wei Lin, and [Yisen Wang](https://yisenwang.github.io/).


STS is a metric that can predict the transferability of LLMs before training. STS identifies shifted dimensions in SAE representations and calculates their correlations with downstream domains. Extensive experiments across multiple models and domains show that STS accurately predicts the transferability of supervised fine-tuning, achieving Pearson correlation coefficients above 0.7 with actual performance changes.
