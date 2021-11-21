using Roots, NLsolve
using ForwardDiff
using BenchmarkTools

"""とりあえず適当にスカラ関数で比較"""
function test_scalar()
	f(x) = 7sin(x) + x^2 - 4
	lineplot(f, -5:0.01:5) |> display
	x₀ = 40.
	#* Roots
	println("== Roots ==")
		(@benchmark find_zero($f, $x₀)) |> display
		find_zero(f, x₀) |> display

	#* NLsolve
	println("\n\n== NLsolve ==")
		f!(y, x) = y[1] = f(x[1])
		(@benchmark nlsolve($f!, [$x₀])) |> display
		nlsolve(f!, [x₀]) |> display

	nothing
end

"""とりあえず適当にベクタ関数で比較"""
function test_vector()
	#: Roots は実1変数の連続スカラ関数しかできないので， 
	#: 独立な2つのスカラ関数のベクトル関数について考える
	g1(x) = (x+3)*(x^3-7)+18
	g2(x) = sin(x*exp(x)-1) / (x^2+1)
	f(x) = [g1(x[1]); g2(x[2])]
	x₀ = [-10.; 0]
	lineplot([g1,g2], -5:0.01:5, ylim=(-8, 5)) |> display
	#* Roots
	println("== Roots ==")
		fs = [g1, g2]
		(@benchmark find_zero.($fs, $x₀)) |> display
		find_zero(g1, x₀[1]) |> display
		find_zero(g2, x₀[2]) |> display

	#* NLsolve
	println("\n\n== NLsolve ==")
		f!(y, x) = y .= f(x)
		(@benchmark nlsolve($f!, $x₀)) |> display
		nlsolve(f!, x₀) |> display

	nothing
end

"""比較はできないが一旦干渉のある2変数ベクトル関数"""
function test_vector_onlyNLsolve()
	#: Roots は実1変数の連続スカラ関数しかできないので， 
	#: 独立な2つのスカラ関数のベクトル関数について考える
	f(x) = [
		(x[1]+3)*(x[2]^3-7)+18
    	sin(x[2]*exp(x[1])-1)
	]
	x₀ = [-10.; 0]
	f!(y, x) = y .= f(x)
	
	#* NLsolve
	println("\n== NLsolve only function ==")
		(@benchmark nlsolve($f!, $x₀)) |> display
		nlsolve(f!, x₀) |> display

	println("\n== NLsolve function with jacobian(inline) ==")
		j1(x)=[
			x[2]^3-7 3*x[2]^2*(x[1]+3)
			x[2]*exp(x[1])*cos(x[2]*exp(x[1])-1) exp(x[1])*cos(x[2]*exp(x[1])-1)
		]
		j1!(y, x) = y .= j1(x)
		(@benchmark nlsolve($f!,  $j1!, $x₀)) |> display
		nlsolve(f!, j1!, x₀) |> display

	println("\n== NLsolve function with jacobian(ForwardDiff) ==")
		j2!(y, x) = ForwardDiff.jacobian!(y, f!, x)
		(@benchmark nlsolve($f!, $j2!, $x₀)) |> display
		nlsolve(f!, j2!, x₀) |> display

	nothing
end


function test_vector_repeat(N)
	calca(γ, lcʳᵉˡ) = begin
		a₀ =  5.00e-3
		c 	=  1.37e-4
		η 	=  5.27e4
		k 	=  2.90
		ρ = c * η * (k-1) / (k-lcʳᵉˡ) * lcʳᵉˡ
		(a₀ + (ρ*γ)^3) / (1 + (ρ*γ)^3)
	end

	@show 𝕒 = rand(N) .* 0.8 .+ 0.1
	@show Lc = rand(N)./5 .+ 1
	γ = 0:0.01:1
	f(a,lc,x) = a-calca(x, lc)
	lineplot(γ, calca.(γ, 0.5), ) |> display

	#* NLsolve
		println("\n\n== NLsolve ==")
		x₀ = one.(𝕒) ./ 2
		f!(y, x) = @. y = f(𝕒, Lc, x)
		begin @benchmark nlsolve($f!, $x₀) end |> display
		x₀ = one.(𝕒)
		nlsolve(f!, x₀) |> display

	##* Roots
		println("\n\n== Roots ==")
		gen = (x->f(a,lc,x) for (a,lc) in zip(𝕒, Lc))
		begin @benchmark find_zero.($gen, (0, 1)|>Ref) end |> display
		find_zero.(gen, Ref((0, 1))) |> display

	#* Roots
		println("\n\n== Roots loop ==")
		begin @benchmark for (a,lc) in zip($𝕒, $Lc)
			find_zero(x -> $f(a, lc, x), (0,1))
		end end |> display
		for (a,lc) in zip(𝕒, Lc)
			find_zero(x -> f(a, lc,x), (0,1)) |> display
		end 

	nothing
end


function test_scalar_repeat()
	calca(γ, lcʳᵉˡ) = begin
		a₀ =  5.00e-3
		c 	=  1.37e-4
		η 	=  5.27e4
		k 	=  2.90
		ρ = c * η * (k-1) / (k-lcʳᵉˡ) * lcʳᵉˡ
		(a₀ + (ρ*γ)^3) / (1 + (ρ*γ)^3)
	end

	a = 0.4
	ga(x) = a - calca(x, 0.5)

	lineplot(ga, 0:0.01:1) |> display

	#* NLsolve
		println("\n\n== NLsolve ==")
		f(x) = ga.(x)
		f!(y, x) = y .= f(x)
		(@benchmark nlsolve($f!, [1.])) |> display
		nlsolve(f!, [1.]) |> display

	#* Roots
		println("== Roots ==")
		(@benchmark find_zero($ga, (0,1))) |> display
		find_zero(f, (0,1)) |> display

	nothing
end