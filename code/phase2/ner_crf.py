"""
ner_crf.py -- linear-chain CRF in pure PyTorch (no pytorch-crf).

Used by the optional CRF head (extra X3). Scores are computed in fp32 whatever the autocast
dtype. Illegal IOB2 transitions get a large negative constant, so the CRF can never learn or
decode an O -> I-X jump.

Shapes: emissions (B, T, C), tags (B, T) int64, mask (B, T) bool where every row is a prefix of
True values (the caller drops the EOS position and right-pads).

Constraint note: transitions O -> I-X / B-X -> I-Y are illegal everywhere; the START of a window
is left free because windows cut through entities (see _start).
"""
import torch
import torch.nn as nn

from ner_eval import transition_constraints

_ILLEGAL = -1e4


class CRF(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        C = 1 + 2 * n_labels
        self.num_tags = C
        self.transitions = nn.Parameter(torch.zeros(C, C))       # [prev, cur]
        self.start_transitions = nn.Parameter(torch.zeros(C))
        self.end_transitions = nn.Parameter(torch.zeros(C))
        allowed, a_start = transition_constraints(n_labels)
        self.register_buffer("trans_pen", torch.where(torch.from_numpy(allowed),
                                                      torch.tensor(0.0), torch.tensor(_ILLEGAL)),
                             persistent=False)
        self.register_buffer("start_pen", torch.where(torch.from_numpy(a_start),
                                                      torch.tensor(0.0), torch.tensor(_ILLEGAL)),
                             persistent=False)

    def _trans(self):
        return self.transitions + self.trans_pen

    def _start(self):
        # NOT constrained: a training window often starts inside an entity, so its first gold
        # tag can legitimately be I-X. The "no I-X at the start" rule only holds at the start
        # of a DOCUMENT, and document-level decoding (ner_eval.viterbi) enforces it there.
        return self.start_transitions

    def neg_log_likelihood(self, emissions, tags, mask, reduction="token_mean"):
        """-log p(tags | emissions). reduction: token_mean (sum / #tokens) | sum | none."""
        em = emissions.float()
        mask = mask.bool()
        B, T, C = em.shape
        trans, start = self._trans(), self._start()
        tags = tags.clamp(min=0)
        # gold path score
        score = start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            m = mask[:, t]
            step = trans[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * m
        lengths = mask.sum(1)
        last = tags.gather(1, (lengths - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last]
        # partition function (forward algorithm)
        alpha = start.unsqueeze(0) + em[:, 0]                       # (B, C)
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        log_z = torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)
        nll = log_z - score
        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        return nll.sum() / lengths.sum().clamp(min=1)

    @torch.no_grad()
    def viterbi(self, emissions, mask):
        """Best paths as python lists (one per row, length = mask row sum)."""
        em = emissions.float()
        mask = mask.bool()
        B, T, C = em.shape
        trans = self._trans()
        score = self._start().unsqueeze(0) + em[:, 0]
        history = []
        for t in range(1, T):
            cand = score.unsqueeze(2) + trans.unsqueeze(0)          # (B, prev, cur)
            best, idx = cand.max(1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), nxt, score)
            history.append(idx)
        score = score + self.end_transitions.unsqueeze(0)
        lengths = mask.sum(1).tolist()
        out = []
        for b in range(B):
            L = lengths[b]
            if L == 0:
                out.append([])
                continue
            tag = int(score[b].argmax())
            path = [tag]
            for t in range(L - 1, 0, -1):
                tag = int(history[t - 1][b, tag])
                path.append(tag)
            out.append(path[::-1])
        return out

    def export(self):
        """numpy transition tensors for document-level Viterbi in ner_eval.decode_scores."""
        return {"transitions": self.transitions.detach().float().cpu().numpy(),
                "start": self.start_transitions.detach().float().cpu().numpy(),
                "end": self.end_transitions.detach().float().cpu().numpy()}
