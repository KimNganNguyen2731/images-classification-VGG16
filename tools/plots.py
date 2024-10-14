import matplotlib.pyplot as plt

def loss_curve(loss_result):
  plt.plot(loss_result)
  plt.title('Loss Curve')
  plt.ylabel('Loss')
  plt.xlabel('Step')
  plt.savefig("loss_curve.png")
  plt.show()
