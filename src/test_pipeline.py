import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.data import Batch, Data

from src.model import FusionModel, GINet
from src.train import clip_loss


def test_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    lincs_dim = 978
    chembl_dim = 1283
    gnn_feat_dim = 256
    embed_dim = 256
    batch_size = 4

    raw_gin = GINet(num_layer=5, emb_dim=300, feat_dim=gnn_feat_dim, drop_ratio=0.0)
    model = FusionModel(
        gnn_model=raw_gin,
        lincs_input_dim=lincs_dim,
        chembl_input_dim=chembl_dim,
        gnn_feat_dim=gnn_feat_dim,
        embed_dim=embed_dim,
    ).to(device)

    # random data generation
    x = torch.randint(0, 3, (20, 2))
    edge_index = torch.randint(0, 20, (2, 40))
    edge_attr = torch.randint(0, 3, (40, 2))
    lincs_batch = torch.randn(batch_size, lincs_dim).to(device)
    chembl_batch = torch.randn(batch_size, chembl_dim).to(device)

    data_list = [
        Data(x=x, edge_index=edge_index, edge_attr=edge_attr) for _ in range(batch_size)
    ]
    batch_graph = Batch.from_data_list(data_list).to(device)

    with profile(
        activities=activities, record_shapes=True, profile_memory=True, with_stack=True
    ) as prof:
        with record_function("model_forward"):
            graph_emb, assay_emb, logit_scale = model(
                batch_graph, lincs_batch, chembl_batch
            )

        with record_function("model_backward"):
            loss = clip_loss(graph_emb, assay_emb, logit_scale)
            loss.backward()

    print("\n---PROFILER RESULTS (sorted by cuda memory)---")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    print("\n---PROFILER RESULTS (sorted by exec time)---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    test_pipeline()
