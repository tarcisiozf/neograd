# NEOGRAD

## FROM SCRATCH
### Lessons learned

After porting the code from Python to Go, I didn't get the expected results, my gradient descent was getting worse and worse.

To fix that first I had to have deterministic results for both implementations, so I pre-trained on python and dumped the parameters in JSON, so I could reuse them on Go. Then I did debugging on both languages, parameter by parameter looking for bugs

That didn't work at scale so I had to build a tool to compare the outputs of both implementations, so I created what I called "memstate" in both![img.png](img.png), which worked similarly to a memory dump, so I could compare the state of the network at any given time

I choose a third language (JS) to rule possible issues floating-point implementations and slowly started increasing the size of inputs to see the differences and once spotted I resorted back to manual debugging 

Finally I was able to catch the bug on my softmax implementation, that was built by porting `np.sum`, not python's builtin `sum` and that was the root cause :)