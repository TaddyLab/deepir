suppressMessages(library(textir))
src <- "yelp"
revs <- read.table(sprintf("data/%s_phrases.txt", src),
	sep="|",quote=NULL, comment="", 
	col.names=c("rev","phrase","y","sample"))

x <- sparseMatrix( 
			i=revs[,1]+1, j=as.numeric(revs[,2]), x=rep(1,nrow(revs)),
			dimnames=list(NULL, levels(revs[,2])) )
x <- x[,colSums(x>0)>5]
y <- as.matrix(sparseMatrix( i=revs[,1]+1, j=revs[,3]+1 )[,2])

levels(revs[,4]) <- c("train","test")
testset <- which(sparseMatrix( i=revs[,1]+1, j=as.numeric(revs[,4])+1 )[,2])

xflat <- rbind(
	colSums(x[-testset,][y[-testset]==0,]),
	colSums(x[-testset,][y[-testset]==1,]))
colnames(xflat) <- colnames(x)
yflat <- c(0,1)

fit <- mnlm(cl=NULL, yflat, xflat, verb=1)

z <- srproj(fit, x, select=100)

fwd <- gamlr(z[-testset,], y[-testset], family="binomial", lambda.min.ratio=1e-4, gamma=1)

yhat <- as.numeric( predict(fwd, z[testset,], type="response") > 0.5 )
cat("MC rate for mnir\n bad: ", mean(yhat[y[testset]==0]), 
	", good: ", 1-mean(yhat[y[testset]==1]), 
	", overall: ", mean(y[testset]!=yhat), "\n")

x <- cBind(z[,1],x)
fitxz = gamlr(x[-testset,], y[-testset], family="binomial", lambda.min.ratio=1e-4)

yhat <- as.numeric(predict(fitxz, x[testset,], type="response") > 0.5)
cat("MC rate with text and SR\n bad: ", mean(yhat[y[testset]==0]), 
	", good: ", 1-mean(yhat[y[testset]==1]), 
	", overall: ", mean(y[testset]!=yhat), "\n")
