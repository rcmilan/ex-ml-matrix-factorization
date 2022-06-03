open Microsoft.ML
open Microsoft.ML.Data
open System
open System.IO
open Microsoft.ML.Trainers

open App.ProductEntry
open App.Prediction

let InputUInt32 (text : string)=
    Console.Write(text)
    let v = Convert.ToUInt32(Console.ReadLine())
    v

let dataPath = Path.Combine(__SOURCE_DIRECTORY__, "Data\\Amazon0302.txt")

// STEP 1: Cria o contexto para ser utilizado durante todo o processo
let mlContext = new MLContext()

// STEP 2: Lê os dados (histórico de compras)
let trainData =
    let columns =
        [|
            TextLoader.Column("Label", DataKind.Single, 0)
            TextLoader.Column("ProductID", DataKind.UInt32, source = [|TextLoader.Range(0)|], keyCount = KeyCount 262111UL)
            TextLoader.Column("ProductID_Copurchased", DataKind.UInt32, source = [|TextLoader.Range(1)|], keyCount = KeyCount 262111UL)
        |]
    mlContext.Data.LoadFromTextFile(dataPath, columns, hasHeader = true, separatorChar = '\t')

// STEP 3: Prepara as configurações do MatrixFactorization trainer
let options = MatrixFactorizationTrainer.Options(MatrixColumnIndexColumnName = "ProductID",
                                                    MatrixRowIndexColumnName = "ProductID_Copurchased",
                                                    LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                                                    LabelColumnName = "Label",
                                                    Alpha = 0.01,
                                                    Lambda = 0.025)

// STEP 4: aplica as configurações
let est = mlContext.Recommendation().Trainers.MatrixFactorization(options)

// STEP 5: Treina o model
let model = est.Fit(trainData)

// STEP 6: faz a predição
let productId = InputUInt32 "Digite o productId do 1º produto: "
let coPurchasedProductId = InputUInt32 "Digite o productId do 2º produto: "

let predictionEngine = mlContext.Model.CreatePredictionEngine<ProductEntry, Prediction>(model)
let prediction = predictionEngine.Predict { ProductID = productId; ProductID_Copurchased = coPurchasedProductId; Label = 0.f }

printfn ""
printfn "Para o ProductID = %d e o CoPurchaseProductID = %d o score é: %f" productId coPurchasedProductId prediction.Score
printf "=============== End of process, hit any key to finish ==============="
Console.ReadKey() |> ignore