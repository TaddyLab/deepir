library(Matrix)

yelp <- read.table("data/yelp_phrases.txt",sep="|",comment="",quote=NULL)

x <- sparseMatrix( 
			i=yelp[,1]+1, j=as.numeric(yelp[,2]), x=rep(1,nrow(yelp)),
			dimnames=list(NULL, levels(yelp[,2])) )
x <- x[,colSums(x>0)>5]
y <- as.numeric(sparseMatrix( i=yelp[,1]+1, j=yelp[,3]+1 )[,2])
testset <- read.table("data/yelp_test.txt")[,1] + 1

library(gamlr)
logistic = gamlr(x[-testset,], y[-testset], lmr=1e-4, family="binomial")

png(file="graphs/yelp_logistic.png", width=12,height=5, units="in", res=360)
plot(logistic)
dev.off()

yhat <- as.numeric(predict(logistic, x[testset,], type="response") > 0.5)
cat("MC rate; bad: ", 1-mean(y[testset][1:10000]==yhat[1:10000]), 
	" good: ", 1-mean(y[testset][10001:20000]==yhat[10001:20000]), "\n")
