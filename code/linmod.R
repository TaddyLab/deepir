suppressMessages(library(Matrix))
args <- commandArgs(TRUE)
src <- args[1]

revs <- read.table(sprintf("data/%s_phrases.txt", src),
	sep="|",quote=NULL, comment="", 
	col.names=c("rev","phrase","y","sample"))

x <- sparseMatrix( 
			i=revs[,1]+1, j=as.numeric(revs[,2]), x=rep(1,nrow(revs)),
			dimnames=list(NULL, levels(revs[,2])) )
x <- x[,colSums(x>0)>5]
y <- as.numeric(sparseMatrix( i=revs[,1]+1, j=revs[,3]+1 )[,2])

levels(revs[,4]) <- c("train","test")
testset <- which(sparseMatrix( i=revs[,1]+1, j=as.numeric(revs[,4])+1 )[,2])

library(gamlr)
fit = gamlr(x[-testset,], y[-testset], family="binomial", lmr=1e-4)

png(file=sprintf("graphs/%s_logistic.png",src), width=12,height=5, units="in", res=360)
plot(fit)
invisible(dev.off())

yhat <- as.numeric(predict(fit, x[testset,], type="response") > 0.5)
cat("MC rate\n bad: ", mean(yhat[y[testset]==0]), 
	", good: ", 1-mean(yhat[y[testset]==1]), 
	", overall: ", mean(y[testset]!=yhat), "\n")

