# BTAD License Evidence Record

Checked on 2026-08-30. This is a provenance record for research preparation, not legal advice.

## Conclusion

The BTAD dataset is identified by its original author repository as licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

The repository's visible label says `CC-BY-SA`, and that label links directly to the Creative Commons `by-sa/4.0/legalcode` page. This link target resolves the version as 4.0.

## Primary evidence chain

1. Original paper-author repository: <https://github.com/pankajmishra000/VT-ADL>
2. Repository README, dataset section: identifies the source as BeanTech srl and labels BTAD `CC-BY-SA`.
3. The README's license hyperlink target: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>
4. Original paper record: <https://arxiv.org/abs/2104.10036>, related IEEE DOI `10.1109/ISIE45552.2021.9576231`.
5. Author-hosted dataset download referenced by the repository: <https://avires.dimi.uniud.it/papers/btad/btad.zip>.

## Local archive corroboration

The locally retained `data/btad_raw/README.txt` repeats:

- `Source: BeanTech srl`
- `License type: CC-BY-SA`
- the VT-ADL citation and author list

Local README record:

- size: 1,228 bytes
- SHA256: `83924497AF8DC9E23295664A834E1F371C74DE5B36068FE65FD7E511B422E395`

The local text does not print the version number, but the original author's live README links its license label to the 4.0 legal code.

## Important distinction

The `LICENSE` file at the repository root is an **MIT License for the VT-ADL software code**. It must not be used as the dataset license. The dataset license is the separate CC BY-SA 4.0 designation in the README's BTAD section.

## Practical handling

- Cite Mishra et al., VT-ADL, ISIE 2021, DOI `10.1109/ISIE45552.2021.9576231`.
- Attribute BeanTech srl as the dataset source where appropriate.
- Include the CC BY-SA 4.0 URL in the dataset/license table.
- If redistributing original or adapted dataset material, preserve attribution, indicate modifications, link the license, and follow ShareAlike requirements.
- The current compact reproducibility package should continue to exclude BTAD images; users can obtain them from the author-hosted source.

