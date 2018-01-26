using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")

lowbound(x, a) = a*(x<a) + (x>=a)*x
function mypool(x)
	p = pool(x)
	m = pool(-x)
	q =-1*(m.>p)
	r = p.>=m
	return (p.*r)+(m.*q)
end

function minpool(x)
	y = 1 ./x
	y = pool(y)
	1 ./ y
end
	

function predict(w, x; clip=false, pdrop=0.5, input_do = 0.0)
    i = 1
		attempt = false
    x = reshape(x, 28, 28, 1, length(x)÷(28*28))
    x = dropout(x, input_do; training= w[1] isa Rec ? true : false)
    if clip 
        x = conv4(sign.(w[i]), x; padding=0) .+w[i+1]
				if attempt
					x = mypool(x)
				else
        	x = pool(x, mode=1)
				end
        x = sign.(x)
        i += 2    
    
        x = conv4(sign.(w[i]), x; padding=0) .+w[i+1]
				if attempt
        	x = mypool(x)
				else
					x = pool(x, mode=1)
				end
        x = sign.(x)
        i += 2    
    
        x = mat(x)
        
        x = sign.(w[i]) * x .+ w[i+1]
        x = sign.(x)
        i += 2 

        return sign.(w[i]) * x .+ w[i+1]
    else 
        scale = w[1] isa Rec ? 1-pdrop : 1D
        μ = conv4(w[1], x; padding = 0) .+ w[2]
        σ² = conv4(1 .- w[1] .* w[1], x .* x; padding = 0)
				σ² = lowbound.(σ², 0.001)
				if attempt
					μ = mypool(μ)
        	σ² = minpool(σ²)
				else
					μ = pool(μ, mode=1)
        	σ² = pool(σ², mode=1)
				end
				x = @. μ/√σ²
				x = 2H.(-x) -1
        N = prod(size(w[3], 1,2,3))
        μ = conv4(w[3], x) .+ w[4]
        σ² = N - conv4(w[3].*w[3], x .* x) 
				σ² = lowbound.(σ², 0.001)
				if attempt
					μ = mypool(μ)
        	σ² = minpool(σ²)
				else
        	μ = pool(μ, mode=1)
        	σ² =pool(σ², mode=1)
				end
				x = @. μ/√σ²
        x = 2H.(-x) - 1
        x = mat(x)
        x = dropout(x, pdrop)

        N = size(w[5], 2)
        μ = w[5]*x .+ w[6]
        σ² = N - (w[5] .* w[5]) * (x.*x)
				σ² = lowbound.(σ², 0.001)
        x = @.  2H(-μ / √σ²) - 1
        x = dropout(x, pdrop)
        
        N = size(w[7], 2)
        μ = w[7]*x .+ w[8]
        σ² = N - (w[7] .* w[7]) * (x.*x)
				σ² = lowbound.(σ², 0.001)
        return @. μ / √σ²
    end
end


losslogH(y) = -sum(logH.(-y)) / size(y, 2)

function loss(w, x, y; pdrop=0.5, input_do = 0.0)
    ŷ = predict(w, x; pdrop=pdrop, input_do = input_do)
    y = onehot!(similar(ŷ), y)
    return losslogH((2 .* y .- 1) .* ŷ)# + 0.01*sum(sum(1-w[i].*w[i])/length(w[i]) for i=1:2:length(w))
end


function build_net(; atype=Array{F})
    w = [
      2rand(5,5,1,20)-1,
      zeros(1,1,20,1),
      
      2rand(5,5,20,50)-1,
      zeros(1,1,50,1),
      
      2rand(500,800)-1,
      zeros(500,1),
      
      2rand(10,500)-1,
      zeros(10,1)
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
        pdrop = 0.5,
        input_do = 0.0
        )

#xtrn = 2xtrn-1
#xtst = 2xtst-1
natural_grad = true
    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    
    acctrn = 0
    acctst = 0
    
    report(epoch) = begin
            dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
            dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
            acctrn = accuracy(w, dtrn, (w,x)->predict(w,x, clip=true))
            acctst = accuracy(w, dtst, (w,x)->predict(w,x, clip=true)) 
            println((:epoch, epoch,
                :trn, accuracy(w, dtrn, (w,x)->predict(w,x)) |> percentage,
                :trn_clip, acctrn  |> percentage,
                :tst, accuracy(w, dtst, (w,x)->predict(w,x))  |> percentage,
                :tst_clip, acctst  |> percentage
            ))

            if reportname != ""
                open(reportname, "a") do f 
                    print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                    for lay = 1:2:length(w)
                        print(f, vecnorm(w[lay])/√length(w[lay]), "\t", mean(1 - w[lay].*w[lay]), "\t")
                    end
                    println(f)
                end
            end

            if verb > 1
                for i=1:2:length(w)
                    n, m = length(w[i]), length(w[i+1])
                    print(" layer $(i÷2+1): W-norm $(round(vecnorm(w[i])/√n, 3))")
                    print(" Θ1-norm $(round(vecnorm(w[i+1])/√m, 3))    std:$(std(convert(Array, abs.(w[i]))))\n")
                end
            end    
        end

    report(0); tic()
    @time for epoch=1:epochs
        #epoch == 10 && setlr!(opt, lr/=2)
        for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
            dw = grad(loss)(w, x, y; pdrop=pdrop, input_do = input_do)
            for i=1:2:length(w)
                dw[i] = (1 - w[i] .* w[i]) .* dw[i]
            end
            update!(w, dw, opt)
						if natural_grad
            	for i=1:2:length(w)
              	w[i] = clip(w[i], 1e-3)
								#w[i] = w[i] +0.001*(vecnorm(w[i])/√length(w[i]) - w[i])
            	end
						end
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 100 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
    return w
end
