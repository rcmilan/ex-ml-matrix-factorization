namespace App

module ProductEntry =

    open Microsoft.ML.Data

    [<CLIMutable>]
    type ProductEntry = 
        {
            [<LoadColumn(0); KeyType(count=262111UL)>]
            ProductID : uint32
            [<LoadColumn(1); KeyType(count=262111UL)>]
            ProductID_Copurchased : uint32
            [<NoColumn>]
            Label : float32
    }