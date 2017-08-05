"""
learning_curve.py 
"""
import matplotlib.pyplot as plt

X = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600]

# ACTION
LRNN_ACT = [0, .678, .736, .892, .902, .939, .942, .953, .955, .956]
JD_ACT = [0, .766, .864, .949, .939, .956, .963, .966, .963, .966]
ID_ACT = [0, .776, .922, .959, .966, .971, .973, .973, .971, .969]

# GOAL
LRNN_GOAL = [0, .649, .711, .763, .842, .825, .842, .842, .851, .842]
JD_GOAL = [0, .491, .728, .561, .728, .728, .845, .845, .853, .852]
ID_GOAL = [0, .447, .570, .561, .711, .754, .767, .801, .798, .798]

plt.figure()
lrnn, = plt.plot(X, LRNN_ACT, 'r-', label='Single-RNN')
jdraggn, = plt.plot(X, JD_ACT, 'g-', label='J-DRAGGN')
idraggn, = plt.plot(X, ID_ACT, 'b-', label='I-DRAGGN')
plt.title('Action-Oriented Test Accuracy vs. # Training Examples')
plt.xlabel('Number of Examples (Combined Action and Goal Samples)')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(handles=[lrnn, jdraggn, idraggn], loc='lower right')
plt.savefig("./action_curve.png")
plt.clf()

lrnn, = plt.plot(X, LRNN_GOAL, 'r-', label='Single-RNN')
jdraggn, = plt.plot(X, JD_GOAL, 'g-', label='J-DRAGGN')
idraggn, = plt.plot(X, ID_GOAL, 'b-', label='I-DRAGGN')
plt.title('Goal-Oriented Test Accuracy vs. # Training Examples')
plt.xlabel('Number of Examples (Combined Action and Goal Samples)')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(handles=[lrnn, jdraggn, idraggn], loc='lower right')
plt.savefig("./goal_curve.png")

