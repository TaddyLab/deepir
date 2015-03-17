suppressMessages(library(Matrix))

imdb <- read.table("data/imdb_phrases.txt",
	sep="|",quote=NULL, comment="", 
	col.names=c("rev","phrase","y","sample"))

x <- sparseMatrix( 
			i=imdb[,1]+1, j=as.numeric(imdb[,2]), x=rep(1,nrow(imdb)),
			dimnames=list(NULL, levels(imdb[,2])) )
x <- x[,colSums(x>0)>5]
y <- as.numeric(sparseMatrix( i=imdb[,1]+1, j=imdb[,3]+1 )[,2])

levels(imdb[,4]) <- c("train","test")
testset <- which(sparseMatrix( i=imdb[,1]+1, j=as.numeric(imdb[,4])+1 )[,2])

library(gamlr)
fit = gamlr(x[-testset,], y[-testset], family="binomial")

png(file="graphs/imdb_logistic.png", width=12,height=5, units="in", res=360)
plot(fit)
invisible(dev.off())

yhat <- as.numeric(predict(fit, x[testset,], type="response") > 0.5)
cat("MC rate\n bad: ", mean(yhat[y[testset]==0]), 
	", good: ", 1-mean(yhat[y[testset]==1]), 
	", overall: ", mean(y[testset]!=yhat), "\n")

