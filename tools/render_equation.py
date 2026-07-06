import matplotlib.pyplot as plt

# Equation which is correct but doesn't work with matplotlib.
# equation = r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V"
# Alternative which works
equation = r"$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$"


plt.text(0.5, 0.5, equation, fontsize=20, ha="center")
plt.axis("off")
# plt.savefig("eq.png", dpi=300, bbox_inches='tight')
plt.show()
