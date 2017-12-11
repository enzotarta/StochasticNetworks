function collect_results(;method_name = "", dataset = "", path = "", preprocessing = "", dropout = 0.0, wall_seed = 1, wall_epochs = 200, lr = "1e-3")
	test_errors = zeros(wall_seed)
	
	if path == ""
		error("No path in input")
	end
	for seed = 1:wall_seed
		resultfile = readdlm(path*join(seed)*".txt")
		test_errors[seed] = mean(resultfile[end-20:end, 3])
	end
	
	if (method_name == "" || dataset == "" || preprocessing == "")
		warn("Incomplete informations available for final report file")
	end
	f = open("final_report.txt", "a")
	print(f, method_name)
	for i = length(method_name)+1:18 print(f, " ") end
	print(f, dataset)
	for i = length(dataset)+1:10 print(f, " ") end
	print(f, preprocessing)
	for i = length(preprocessing)+1:14 print(f, " ") end
	print(f, lr)
	for i = length(lr)+1:6 print(f, " ") end
	print(f, wall_epochs)
	for i = length(string(wall_epochs))+1:10 print(f, " ") end
	print(f, string(dropout))
	for i = length(dropout)+1:8 print(f, " ") end
	print(f, wall_seed)
	for i = length(string(wall_seed))+1:6 print(f, " ") end
	print(f, mean(test_errors))
	for i = length(string(mean(test_errors)))+1:20 print(f, " ") end
	println(f, std(test_errors)/sqrt(wall_seed))
	#println(f, method_name,"\t", dataset, "\t\t", preprocessing, "\t\t\t\t\t\t", lr, "\t", wall_epochs, "\t", dropout, "\t\t\t\t", wall_seed, "\t\t", mean(test_errors), "\t", std(test_errors)/sqrt(wall_seed))
	info("Method:", method_name,"   DATASET:", dataset, "   preprocessing:", preprocessing, "  dropout:", dropout, "   wall seed:", wall_seed, "    mean:", mean(test_errors), "  std:", std(test_errors)/sqrt(wall_seed))
	close(f)
end
