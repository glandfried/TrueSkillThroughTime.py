all: history.csv summary_filtered.csv

ChessAnalysis.zip:
	#  Blog 2008
	#https://docs.microsoft.com/en-us/archive/blogs/apg/trueskill-through-time
	# Blog 2012
	#https://docs.microsoft.com/en-us/archive/blogs/dsyme/updated-version-of-trueskill-through-time-bayesian-inference-code
	wget https://msdnshared.blob.core.windows.net/media/MSDNBlogsFS/prod.evol.blogs.msdn.com/CommunityServer.Components.PostAttachments/00/10/29/54/02/ChessAnalysis.zip
	unzip ChessAnalysis.zip

	#sudo apt-get install fsharp
	#xbuild /p:Configuration=Release ChessAnalysis.fsproj
	#mono bin/Release/ChessAnalysis.exe -out output.csv SmallChessBase.csv

summary_filtered.csv:
	wget https://github.com/glandfried/centrality-ogs-dataset/releases/download/v0.0/summary_filtered.zip
	unzip summary_filtered.zip
	rm summary_filtered.zip 

history.csv:
	wget https://github.com/glandfried/tennis_atp/releases/download/atp/history.csv.zip
	unzip history.csv.zip
	rm history.csv.zip



