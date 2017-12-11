using Knet
using MLDatasets

const F = Float32
include("../../utility/common.jl")

# all cnn are "same" 3x3  
NET = [128,128, 'M', 256, 256, 'M',
  512, 512,  'M', 'F', [1024]]


function predict(w, x, bmom; pdrop=0.5, input_do = 0.0)
    i = 1
    x = reshape(x, 32, 32, 3, length(x)÷(32*32*3))
    x = dropout(x, input_do; training= w[1] isa Rec ? true : false)
 
    for f in NET
        f == 'F' && break
        if f != 'M'
            x = conv4(w[i], x; padding=1)
            x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
            x = relu.(x)
            i += 3
        else
            x = pool(x)
        end
    end
    x = mat(x)
    for f in NET[end]
        x = w[i] * x
        x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
        x = relu.(x)
        x = dropout(x, pdrop)
        i += 3
    end

    x = w[i] * x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i+= 3
    return x
end
 
loss(w, x, y, bmom; pdrop=0.5) = nll(predict(w, x, bmom; pdrop=pdrop, input_do = input_do), y)

function build_net(; atype=Array{F})
    w = []
    f0 = 3
    for f in NET
        f == 'M' && continue
        f == 'F' && break
        push!(w, xavier(3,3,f0, f))
        push!(w, ones(1,1,f,1))
        push!(w, zeros(1,1,f,1))
        f0 = f
    end
    side = 32 ÷ 2^count(x->x=='M', NET)
    f0 = f0 * side^2
    for f in NET[end]
        push!(w, xavier(f,f0))
        push!(w, ones(f,1))
        push!(w, zeros(f,1))
        f0 = f
    end
    push!(w, xavier(10,f0))
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
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
				reportname = "",
        pdrop = 0.,  #dropout probability
        input_do = 0.0
        )

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = [BatchMoments() for _=1:length(w)÷3]

    acctrn = 0
    acctst = 0
    dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
    dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
    report(epoch) = begin
            acctrn = accuracy(w, dtrn, (w,x) ->predict(w,x,bmom))
            acctst = accuracy(w, dtst, (w,x) ->predict(w,x,bmom))
            println((:epoch, epoch,
                :trn, acctrn |> percentage,
                :tst, acctst |> percentage
                ))
						
            if reportname != ""
                open(reportname, "a") do f 
                    print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                    for lay = 1:3:length(w)
                        print(f, vecnorm(w[lay])/√length(w[lay]), "\t")
                    end
                    println(f)
                end
            end
        end


    report(0); tic()
    @time for epoch=1:epochs
        for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
            dw = grad(loss)(w, x, y, bmom; pdrop=pdrop, input_do = input_do)
            update!(w, dw, opt)
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 200 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
    return w, acctst
end

