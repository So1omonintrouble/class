using Plots
gr()

@userplot CirclePlot
@recipe function f(cp::CirclePlot)
    x, y, i = cp.args
    n = length(x)
    inds = circshift(1:n, 1 - i)
    linewidth --> range(0, stop = 10, length = n)
    alpha --> range(0, stop = 1, length = n)
    aspect_ratio --> 1
    label --> false
    x[inds], y[inds]
end

n = 400
balabala = range(0, stop = 2π, length = n)
x = 16sin.(balabala).^3
println(x)

y = 13cos.(balabala) .- 5cos.(2balabala) .- 2cos.(3balabala) .- cos.(4balabala)

anim = @animate for i ∈ 1:n
    circleplot(x, y, i, line_z = 1:n, cbar = false, c = :reds, framestyle = :none)
end when i > 40 && mod1(i, 10) == 5

gif(anim, "to_you.gif")

# http://docs.juliaplots.org/latest/
