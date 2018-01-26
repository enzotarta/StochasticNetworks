using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")

# all cnn are "same" 3x3  
NET = [128,128, 'M', 256, 256, 'M',
  512, 512,  'M', 'F', [1024, 1024]]

#sclamp(x) = (x.>1)*1 + (x.<-1).*(-1) + (abs.(x).<1).*x
lowbound(x, a) = a*(x<a) + (x>=a)*x

function predict(w, x, bmom; clip=false, pdrop=0.5, input_do = 0.0)
	i = 1
	x = reshape(x, 32,32,3, length(x)÷(32*32*3))
	x = dropout(x, input_do; training= w[1] isa Rec ? true : false)
	if clip
		for idx = 1:length(NET)
    	NET[idx] == 'F' && break
      if NET[idx] != 'M'
	      x = conv4(sign.(w[i]), x; padding=1).+w[i+2]
        if NET[idx+1] == 'M'
        	x = pool(x, mode=1)
       	end
        #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
        x = sign.(x)
        i += 3
       end
    end
    x = mat(x)
    for f in NET[end]
    	x = sign.(w[i]) * x .+ w[i+2]
      #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
      x = sign.(x)
      i += 3    
    end
    x = sign.(w[i]) * x .+ w[i+2]
    #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i+= 3
    return x
	else
		for idx = 1:length(NET)
			NET[idx] == 'F' && break
			if NET[idx] != 'M'
				N = length(w[i])÷ size(w[i], 4)
				μ = conv4(w[i], x; padding=1).+ w[i+2]
				σ² = i == 1 ? conv4(1 .- w[i] .* w[i], x .* x; padding=1) : N - conv4(w[i].*w[i], x .*x; padding=1)
				if NET[idx+1] == 'M'
					μ = pool(μ, mode=1)
        	σ² = pool(σ², mode=1)
				end
				σ² = lowbound.(σ², 0.0001)
				x = @. -μ / √(σ²)
				#x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
				x = @.  2*H(x) - 1
				i += 3
      end
    end
    x = mat(x)
    for f in NET[end]
			#x = dropout(x, pdrop)
				N = size(w[i], 2)
        μ = w[i] * x .+ w[i+2]
        σ² = N .- (w[i] .* w[i]) * (x .* x)
				σ² = lowbound.(σ², 0.0001)
        x = @. -μ / √(σ²)
        #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
        x = @.  2H(x) - 1
        i += 3 

    end
		N = size(w[i], 2)
    μ = w[i] * x .+ w[i+2]
    σ² = N .- (w[i] .* w[i]) * (x .* x)
		σ² = lowbound.(σ², 0.0001)
    x = @. μ / √(σ²)
    #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i+= 3
    return H.(-x)
	end
end


losslogH(y) = -sum(logH.(-y)) / size(y, 2)

#=function loss(w, x, y, bmom; pdrop=0.5, input_do = 0.0)
    ŷ = predict(w, x, bmom; pdrop=pdrop, input_do = input_do)
    y = onehot!(similar(ŷ), y)
    return losslogH((2 .* y .- 1) .* ŷ)
end=#

function loss(w, x, y, bmom; clip = false, pdrop=0.0, input_do = 0.0, ρ=0.5, ϵ=0.000001)
  ŷ = predict(w, x, bmom; clip=clip, pdrop=pdrop, input_do = input_do)
  nm = onehot!(similar(ŷ), y)
  ŷ = ŷ ./(1.0 + ϵ - ŷ)
  ŷ -= ρ .* ŷ .* nm
	ŷ = ŷ ./ maximum(ŷ, 1)
  s = _sum_loss(nm .* ŷ) - _sum_loss(ŷ)
  return -s
end

function _sum_loss(ŷ)
  sum(log.(sum(ŷ, 1)))
end

#loss(w, x, y, bmom; input_do = 0.2, pdrop = 0.0) = nll(predict(w, x, bmom), y)

function build_net(; atype=Array{F})
    w = []
    f0 = 3
    for f in NET
        f == 'M' && continue
        f == 'F' && break
        push!(w, 2rand( 3,3,f0, f)-1)
        push!(w, ones(1,1,f,1))
        push!(w, zeros(1,1,f,1))
        f0 = f
    end
    side = 32 ÷ 2^count(x->x=='M', NET)
    f0 = f0 * side^2
    for f in NET[end]
        push!(w, 2rand( f,f0)-1)
        push!(w, ones(f,1))
        push!(w, zeros(f,1))
        f0 = f
    end
    push!(w, 2rand(10,f0)-1)
    push!(w, ones(10,1))
    push!(w, zeros(10,1))
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
xtrn = 2xtrn - 1
xtst = 2xtst - 1
#xtrn = xtrn[:,:,:,1:1000]
#ytrn = ytrn[1:1000]
#xtst = xtst[:,:,:,1:100]
#ytst = ytst[1:100]

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = [BatchMoments() for _=1:(length(w)÷3)]
    
    acctrn = 0
    acctst = 0
             dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
            dtst = minibatch(xtst, ytst, batchsize; xtype=atype)   
    report(epoch) = begin
            acctrn = accuracy(w, dtrn, (w,x)->predict(w,x, bmom; clip=true))
            acctst = accuracy(w, dtst, (w,x)->predict(w,x, bmom; clip=true)) 
            println((:epoch, epoch,
                :trn, accuracy(w, dtrn, (w,x)->predict(w,x, bmom)) |> percentage,
                :trn_clip, acctrn  |> percentage,
                :tst, accuracy(w, dtst, (w,x)->predict(w,x,bmom))  |> percentage,
                :tst_clip, acctst  |> percentage
            ))

            if reportname != ""
                open(reportname, "a") do f 
                    print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                    for lay = 1:3:length(w)-2
                        print(f, vecnorm(w[lay])/√length(w[lay]), "\t", mean(1 - w[lay].*w[lay]), "\t")
                    end
                    println(f)
                end
            end

            if verb > 1
                for i=1:3:length(w)-2
                    n, m, l = length(w[i]), length(w[i+1]), length(w[i+2])
                    print(" layer $(i÷3+1): W-norm $(round(vecnorm(w[i])/√n, 3))\n")
                   # print(" batch-norm $(round(vecnorm(w[i+1])/√m, 3))    $(round(vecnorm(w[i+2])/√l, 3))\n")
                end
            end    
        end

    #report(0);
 tic()
    @time for epoch=1:epochs
        # epoch == 10 && setlr!(opt, lr/=10)
        for (x, y) in minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
            #info(loss(w, x, y, bmom; pdrop=pdrop, input_do = input_do))
 
            dw = grad(loss)(w, x, y, bmom; pdrop=pdrop, input_do = input_do)
#info(loss(w, x, y, bmom; pdrop=pdrop, input_do = input_do))   
         for i=1:3:length(w)-2
               dw[i] = (1 - w[i] .* w[i]) .* dw[i]
            end
            update!(w, dw, opt)
            for i=1:3:length(w)-2
            	w[i] = clip(w[i], 1e-2)
							#w[i] = w[i] + 0.001*(mean(w[i])-w[i])	
            end
#ŷ = convert(Array, predict(w, x, bmom))
#warn((sum((maximum(ŷ, 1).==ŷ).* convert(Array, onehot!(similar(ŷ), y)))/length(y)))
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 100 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
end
