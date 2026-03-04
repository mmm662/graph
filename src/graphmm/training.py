from __future__ import annotations
from typing import List, Optional, Tuple
import os, random
import torch
from tqdm import tqdm

from graphmm.datasets.trajectory_provider import pad_batch, PADDING_ID, Sample, build_transition_graph
from graphmm.models.graphmm_corrector import GraphMMCorrector, decode_argmax, confidence_gate_sequence
from graphmm.utils.graph import token_seq_accuracy, path_feasibility_rate, build_adj_list, k_hop_neighbors

def train_loop(
    model: GraphMMCorrector,
    graph_batch,
    train_samples: List[Sample],
    valid_samples: List[Sample],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    run_dir: str,
    k_hop: int,
    top_r_train: int,
    top_r_decode: int,
    use_crf: bool,
    ss_start: float = 1.0,
    ss_end: float = 1.0,
    ss_mode: str = "linear",
    traj_graph_source: str = "mixed",
    min_correction_confidence: float = 0.0,
    min_correction_logit_gain: float = 0.0,
    eval_apply_gate: bool = False,
    crf_train_loss: str = "ce",
):
    crf_train_loss = str(crf_train_loss).lower()
    if crf_train_loss not in {"ce", "crf"}:
        raise ValueError(f"crf_train_loss must be one of ['ce', 'crf'], got: {crf_train_loss}")

    os.makedirs(run_dir, exist_ok=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    node_num_feat = graph_batch.node_num_feat.to(device)
    floor_id = graph_batch.floor_id.to(device)
    edge_index = graph_batch.edge_index.to(device)
    edge_attr = graph_batch.edge_attr.to(device)

    road_adj_list = build_adj_list(graph_batch.num_nodes, edge_index)
    allowed_prev = k_hop_neighbors(road_adj_list, k=k_hop) if use_crf else None

    def build_traj_graph_from_samples(samples: List[Sample]) -> Tuple[torch.Tensor, torch.Tensor]:
        source = str(traj_graph_source).lower()
        if source not in {"pred", "true", "mixed"}:
            raise ValueError(f"traj_graph_source must be one of ['pred', 'true', 'mixed'], got: {traj_graph_source}")

        seqs = []
        for s in samples:
            if source == "pred":
                seqs.append(s.pred)
            elif source == "true":
                seqs.append(s.true)
            else:
                seqs.append(s.pred)
                seqs.append(s.true)

        # edges with counts
        ecount = build_transition_graph(seqs, directed=True, min_count=1)
        if not ecount:
            # dummy self-loop
            idx = torch.arange(graph_batch.num_nodes, device=device)
            edge_idx = torch.stack([idx, idx], dim=0)
            w = torch.ones(idx.numel(), device=device)
            return edge_idx, w
        src = torch.tensor([u for (u,v,c) in ecount], dtype=torch.long, device=device)
        dst = torch.tensor([v for (u,v,c) in ecount], dtype=torch.long, device=device)
        w = torch.tensor([float(c) for (u,v,c) in ecount], dtype=torch.float32, device=device)
        # normalize weights by mean
        w = w / w.mean().clamp(min=1e-6)
        edge_idx = torch.stack([src,dst], dim=0)
        return edge_idx, w

    def run_eval(samples: List[Sample]):
        model.eval()
        raw_all=[]; pred_all=[]; gated_all=[]; all_g=[]
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                pred_seqs=[s.pred for s in batch]
                true_seqs=[s.true for s in batch]
                pred_pad, lengths = pad_batch(pred_seqs, pad_id=PADDING_ID)
                pred_pad = pred_pad.to(device); lengths = lengths.to(device)

                traj_edge_index, traj_edge_weight = build_traj_graph_from_samples(batch)
                unary_logits, H_R = model.forward_unary(
                    pred_pad, lengths,
                    node_num_feat=node_num_feat,
                    floor_id=floor_id,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    traj_edge_index=traj_edge_index,
                    traj_edge_weight=traj_edge_weight,
                    teacher_forcing=None
                )
                if use_crf:
                    out=[]
                    for bi in range(len(batch)):
                        L = int(lengths[bi].item())
                        unary_i = unary_logits[bi,:L,:]
                        path = model.crf.viterbi_one(unary_i, H_R, allowed_prev, top_r=top_r_decode)
                        out.append(path)
                else:
                    out = decode_argmax(unary_logits, lengths)

                out_gated = [
                    confidence_gate_sequence(
                        raw_seq=pred_seqs[bi],
                        corrected_seq=out[bi],
                        unary_logits=unary_logits[bi, :int(lengths[bi].item()), :],
                        min_confidence=min_correction_confidence,
                        min_logit_gain=min_correction_logit_gain,
                    )
                    for bi in range(len(out))
                ]

                raw_all.extend(pred_seqs)
                pred_all.extend(out)
                gated_all.extend(out_gated)
                all_g.extend(true_seqs)

        raw_tok, _ = token_seq_accuracy(raw_all, all_g)
        pred_tok, pred_seq = token_seq_accuracy(pred_all, all_g)
        gated_tok, gated_seq = token_seq_accuracy(gated_all, all_g)
        final_pred = gated_all if eval_apply_gate else pred_all
        feas = path_feasibility_rate(final_pred, road_adj_list, k_hop=max(1,k_hop))

        total_tokens = 0
        changed_tokens = 0
        for raw, corr in zip(raw_all, final_pred):
            L = min(len(raw), len(corr))
            total_tokens += L
            changed_tokens += sum(int(a != b) for a, b in zip(raw[:L], corr[:L]))
        change_ratio = (changed_tokens / total_tokens) if total_tokens > 0 else 0.0

        return {
            "raw_tok": raw_tok,
            "pred_tok": pred_tok,
            "pred_seq": pred_seq,
            "gated_tok": gated_tok,
            "gated_seq": gated_seq,
            "final_tok": gated_tok if eval_apply_gate else pred_tok,
            "final_seq": gated_seq if eval_apply_gate else pred_seq,
            "feas": feas,
            "changed": change_ratio,
        }


    def teacher_forcing_ratio(epoch_idx: int) -> float:
        if epochs <= 1:
            return float(ss_end)
        start = float(ss_start)
        end = float(ss_end)
        t = (epoch_idx - 1) / max(epochs - 1, 1)
        if ss_mode == "linear":
            ratio = start + (end - start) * t
        else:
            ratio = end
        return float(max(0.0, min(1.0, ratio)))

    best = -1.0
    for ep in range(1, epochs+1):
        model.train()
        random.shuffle(train_samples)
        total_loss=0.0; total_cnt=0

        traj_edge_index, traj_edge_weight = build_traj_graph_from_samples(train_samples)

        for i in tqdm(range(0, len(train_samples), batch_size), desc=f"epoch {ep}"):
            batch = train_samples[i:i+batch_size]
            pred_seqs=[s.pred for s in batch]
            true_seqs=[s.true for s in batch]
            pred_pad, lengths = pad_batch(pred_seqs, pad_id=PADDING_ID)
            true_pad, _ = pad_batch(true_seqs, pad_id=PADDING_ID)

            pred_pad = pred_pad.to(device); true_pad = true_pad.to(device); lengths = lengths.to(device)

            tf_ratio = teacher_forcing_ratio(ep)
            if tf_ratio >= 1.0:
                tf_in = true_pad
            elif tf_ratio <= 0.0:
                tf_in = pred_pad
            else:
                tf_mask = (torch.rand_like(true_pad, dtype=torch.float32) < tf_ratio) & (true_pad != PADDING_ID)
                tf_in = torch.where(tf_mask, true_pad, pred_pad)

            unary_logits, H_R = model.forward_unary(
                pred_pad, lengths,
                node_num_feat=node_num_feat,
                floor_id=floor_id,
                edge_index=edge_index,
                edge_attr=edge_attr,
                traj_edge_index=traj_edge_index,
                traj_edge_weight=traj_edge_weight,
                teacher_forcing=tf_in
            )

            if use_crf and crf_train_loss == "crf":
                loss = torch.tensor(0.0, device=device)
                for bi in range(len(batch)):
                    L = int(lengths[bi].item())
                    unary_i = unary_logits[bi,:L,:]
                    gold_i = true_seqs[bi]
                    loss = loss + model.crf.nll_one(unary_i, gold_i, H_R, allowed_prev, top_r=top_r_train, train_mode=True)
                loss = loss / max(len(batch),1)
            else:
                B, Lm, N = unary_logits.shape
                loss = torch.nn.functional.cross_entropy(unary_logits.view(B*Lm, N), true_pad.view(B*Lm), ignore_index=PADDING_ID)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * len(batch)
            total_cnt += len(batch)

        train_loss = total_loss / max(total_cnt,1)
        if valid_samples:
            em = run_eval(valid_samples)
        else:
            em = {"raw_tok":0.0,"pred_tok":0.0,"pred_seq":0.0,"gated_tok":0.0,"gated_seq":0.0,"final_tok":0.0,"final_seq":0.0,"feas":0.0,"changed":0.0}

        print(
            f"[epoch {ep:02d}] tf_ratio={teacher_forcing_ratio(ep):.3f} traj_graph={str(traj_graph_source).lower()} "
            f"conf_gate={min_correction_confidence:.2f} gain_gate={min_correction_logit_gain:.2f} "
            f"eval_gate={int(eval_apply_gate)} crf_loss={crf_train_loss} loss={train_loss:.4f} "
            f"raw_tok={em['raw_tok']:.3f} pred_tok={em['pred_tok']:.3f} gated_tok={em['gated_tok']:.3f} "
            f"final_tok={em['final_tok']:.3f} final_seq={em['final_seq']:.3f} changed={em['changed']:.3f} feas@k={em['feas']:.3f}"
        )

        score = em["final_tok"] + 0.1 * em["feas"]
        if score > best:
            best = score
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "val_tok": em["final_tok"], "val_seq": em["final_seq"], "feas": em["feas"], "raw_tok": em["raw_tok"], "pred_tok": em["pred_tok"], "gated_tok": em["gated_tok"], "changed": em["changed"]},
                os.path.join(run_dir, "checkpoint.pt")
            )
