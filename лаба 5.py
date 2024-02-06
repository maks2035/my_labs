import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#####
C = 1
T = 1
L = 1 
dL = 0.1 
#####

def f(t):
   answer = -1200*np.sin(4*(t*t+1)) + t*t # если А*синус + В*косинус и  и А и/или В достаточно большие то поврехность "скачет" в центре 
   return answer 

n = int(L/dL)

x = y = np.arange(0.0, L + dL, dL)
t = 0

z = [[30]*(n+1)]*(n)
z.append([100]*(n+1))
z = np.transpose(z)
X, Y = np.meshgrid(x, y)

e = 100*C*C #как я понял это какая-то скорость переноса 
dt = C*dL*dL/(2*e*dL) # условние Куранта-...


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = LinearSegmentedColormap.from_list ('red_blue', ['b','g','y', 'r'], 256)


ax.plot_surface(X, Y, np.array(z), cmap = cmap)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.view_init(elev=40, azim=-120)
blue_proxy = plt.Rectangle((0, 0), 0.1, 0.1, fc="white")
plt.legend([blue_proxy],[''.join(['t=',str('%.4f' % t)])], loc='best', fontsize = 14)
plt.pause(dt)
ax.cla()

while t < T:
   
   U = z.copy()
   for j in np.arange(1, n, 1):
      for i in np.arange(0, n+1, 1):
         if(i == n):
            U[i][j] = z[i][j] + dt*f(t) + (z[i][j] - 2 * z[i][j] + z[i-1][j] + z[i][j+1]- 2*z[i][j] + z[i][j-1]) * dt*C/(dL*dL)
         else:
            U[i][j] = z[i][j] + dt*f(t) + (z[i+1][j] - 2 * z[i][j] + z[i-1][j] + z[i][j+1]- 2*z[i][j] + z[i][j-1]) * dt*C/(dL*dL)
   ax.plot_surface(X, Y, np.array(U),  cmap = cmap)

   ax.set_xlabel('X-axis')
   ax.set_ylabel('Y-axis')
   ax.set_zlabel('Z-axis')
   ax.view_init(elev=40, azim=-120)
   plt.pause(dt)
   ax.cla()
   t = t + dt
   z = U.copy()
   blue_proxy = plt.Rectangle((0, 0), 0.1, 0.1, fc="white")
   plt.legend([blue_proxy],[''.join(['t=',str('%.4f' % t)])], loc='best', fontsize = 14)

ax.plot_surface(X, Y, np.array(z), cmap = cmap)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.view_init(elev=40, azim=-120)
blue_proxy = plt.Rectangle((0, 0), 0.1, 0.1, fc="white")
plt.legend([blue_proxy],[''.join(['t=',str('%.4f' % t)])], loc='best', fontsize = 14)
plt.show()
