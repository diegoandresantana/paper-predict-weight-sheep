# Se precisar carregar pacotes adicionais, siga os exemplos abaixo 
#install.packages("psych")
#install.packages("ggplot2")
library("psych")
library("ggplot2")

# Ler os dados de um arquivo (interagindo com o usuario)
dados <- read.table(file.choose(),header=TRUE , sep = ";")

# Mostra boxplots das Classificadoress, lado a lado, em relação ao PCC
dados$Regressors <- as.factor(dados$Regressors)
bp <- ggplot(dados, aes(x=Regressors, y=Performace,fill=Regressors)) + 
  geom_boxplot()+
  labs(title="Boxplot of R² for Regressors",x="Regressors", y = "R²")
bp + theme_classic()

# Cria a tabela ANOVA 
dados.anova <- aov(dados$Performace ~ dados$Regressors)

# Mostra a tabela ANOVA
summary(dados.anova)

# Realiza e mostra os resultados de um pós-teste usando Tukey
tukey <- TukeyHSD(dados.anova)
tukey

par(mar=c(8,8,8,8))#qtd tecnica, qtd dobras, qtd tecnica, qtd tecnica
plot(tukey , las=1 , col="brown" )