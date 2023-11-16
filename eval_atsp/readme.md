# Evaluation for ATSP

Please refer to [MatNet](https://github.com/yd-kwon/MatNet) to extract the provided ATSP checkpoints (in their repo) first and place them in `./eval_atsp/result/`.

```bash
cd eval_atsp # set eval_atsp as your working directory

python ATSProblemDef.py # generate test instances

python test_glop.py 150 # test GLOP for STSP 150, similarly for 250, and 1000.

python test_matnet.py 150 # test matnet; please also refer to the original repository provided below
```

---

**References**

[Kwon, Y. D., Choo, J., Yoon, I., Park, M., Park, D., & Gwon, Y. (2021). Matrix encoding networks for neural combinatorial optimization. Advances in Neural Information Processing Systems, 34, 5138-5149.](https://github.com/yd-kwon/MatNet)



