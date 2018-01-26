using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")


function predict(w, x, bmom; pdrop=0.5)
    i = 1
    x = reshape(x, 28, 28, 1, length(x)÷(28*28))
 
    μ = conv4(w[i], x; padding=0)
    σ² = conv4(1 .- w[i] .* w[i], x .* x; padding=0)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = pool(x; mode = 1)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    i += 3

    μ = conv4(w[i], x, padding=0)
    σ² = conv4(1 .- w[i] .* w[i], x .* x, padding=0)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = pool(x; mode=1)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    i += 3
    
    x = mat(x)
    x = dropout(x, pdrop)
    μ = w[i] * x
    σ² = (1 .- w[i] .* w[i]) * (x .* x)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    i += 3

    μ = w[i] * x
    σ² = (1 .- w[i] .* w[i]) * (x .* x)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i+= 3
    
    return x
end

loss(w, x, y, bmom; pdrop=0.5, input_do = 0.0) = nll(predict(w, dropout(x, input_do; training=true), bmom; pdrop=0.5), y) + sum(sum(1-w[i].*w[i])/length(w[i]) for i=1:3:length(w))/(length(w)÷3)

binarize(w) = [i%3==1 ? sign.(w[i]) : w[i] for i=1:length(w)]

function inoutmag(w, x, bmom; pdrop=0.0)
    i = 1
    x = reshape(x, 28, 28, 1, length(x)÷(28*28))
    j = 1
    x_avg = zeros(length(w)+1)

    xtemp= reshape(x, length(x)÷size(x, 4), size(x, 4))
    x_avg[j] = mean(sum(xtemp.*xtemp,1)/(size(xtemp, 1)))
		j+=1

    μ = conv4(w[i], x; padding=0)
    σ² = conv4(1 .- w[i] .* w[i], x .* x; padding=0)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = pool(x)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i += 3
    xtemp= reshape(x, length(x)÷size(x, 4), size(x, 4))
    x_avg[j] = mean(sum(xtemp.*xtemp,1)/(size(xtemp, 1)))
		j+=1
x=fSign.(x)
    μ = conv4(w[i], x, padding=0)
    σ² = conv4(1 .- w[i] .* w[i], x .* x, padding=0)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = pool(x)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i += 3
    xtemp= reshape(x, length(x)÷size(x, 4), size(x, 4))
    x_avg[j] = mean(sum(xtemp.*xtemp,1)/(size(xtemp, 1)))
		j+=1
x = fSign.(x)
    x = mat(x)

    μ = w[i] * x
    σ² = (1 .- w[i] .* w[i]) * (x .* x)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i += 3
    
    x_avg[j] = mean(sum(x.*x,1)/(size(x, 1)))
		j+=1
x = fSign.(x)
    μ = w[i] * x
    σ² = (1 .- w[i] .* w[i]) * (x .* x)
    x = μ .+ randn!(similar(μ)) .* sqrt.(σ²)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i+= 3

    x_avg[j] = mean(sum(x.*x,1)/(size(x, 1)))
    return x_avg
end

function build_net(; atype=Array{F})
    w = [
      2rand(5,5,1,20)-1,
      ones(1,1,20,1),
      zeros(1,1,20,1),
      
      2rand(5,5,20,50)-1,
      ones(1,1,50,1),
      zeros(1,1,50,1),
      
      2rand(500,800)-1,
      ones(500,1),
      zeros(500,1),
      
      2rand(10,500)-1,
      ones(10,1),
      zeros(10,1),
    ]
    return map(a->convert(atype,a), w)
end

function main(xtrn, ytrn, xtst, ytst;
        seed = -1,
        batchsize = 200,
        lr = 1e-3,
        epochs = 200,
        infotime = 1,  # report every `infotime` epochs
        reportname = "",
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
        verb = 2,
				input_do = 0.0,
        pdrop = 0.5
        )

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = [BatchMoments() for _=1:(length(w)÷3)]
    
    acctrn = 0
    acctst = 0
    
    report(epoch) = begin
            dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
            dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
            bw = binarize(w) 
            acctrn = accuracy(bw, dtrn, (w,x)->predict(w,x, bmom))
            acctst = accuracy(bw, dtst, (w,x)->predict(w,x, bmom)) 
            println((:epoch, epoch,
                :trn, accuracy(w, dtrn, (w,x)->predict(w,x, bmom)) |> percentage,
                :trn_clip, acctrn  |> percentage,
                :tst, accuracy(w, dtst, (w,x)->predict(w,x,bmom))  |> percentage,
                :tst_clip, acctst  |> percentage
            ))
#=counter = 0
          stat = zeros(length(w)+1)
          for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=false, xtype=atype)
            stat .+= inoutmag(w, x, bmom)
            counter +=1
          end
          for i=1:5
            info("L",i-1,":", stat[i]/counter)
          end
=#
            if reportname != ""
                open(reportname, "a") do f 
                    print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                    for lay = 1:3:length(w)
                        print(f, vecnorm(w[lay])/√length(w[lay]), "\t", mean(1 - w[lay].*w[lay]), "\t")
                    end
  #=                  for i = 1:5
                      print(f, stat[i]/counter, "\t")
                    end=#
                    println(f)
      
                end
            end

            if verb > 1
                for i=1:3:length(w)
                    n, m = length(w[i]), length(w[i+1])
                    print(" layer $(i÷3+1): W-norm $(round(vecnorm(w[i])/√n, 3))")
                    print(" Θ1-norm $(round(vecnorm(w[i+1])/√m, 3))")
                    print(" Θ2-norm $(round(vecnorm(w[i+2])/√m, 3))\n")
                end
            end    
        end

    report(0); tic()
    @time for epoch=1:epochs
        for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=false, xtype=atype)
            dw = grad(loss)(w, x, y, bmom; pdrop=pdrop, input_do = input_do)
            update!(w, dw, opt)
            for i=1:3:length(w)
                w[i] = clip(w[i], 1e-2)
            end
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 100 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
    return w
end
