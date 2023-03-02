using Plots,Plotly

x = 0:0.01:2pi
y = 10cos.(x)
plt = Plotly.plot(x,y)
display(plt)

y2 = 10sin.(x)
plt2 = Plotly.plot(x,y2)
display(plt2)
