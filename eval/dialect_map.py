# coding=utf-8
"""`lang_code` -> the key used by FormoG2P's `g2p_250402.csv`.

The CSV keys a row by `Language` + `Dialect`, both in Chinese, joined with "_"
(a language with no dialect split uses the language alone). Our manifests carry
`lang_code` instead, so the two have to be bridged by hand. All 42 rows of the
CSV correspond one-to-one with the 42 lang_codes in the corpus.

Note 霧臺 vs 霧台: the CSV writes 魯凱_霧台 while the official dialect name is
霧臺魯凱語. The value here follows the CSV, since it is the lookup key.
"""

LANG_CODE_TO_G2P = {
    "ami-x-iams": "阿美_南勢",
    "ami-x-skl": "阿美_秀姑巒",
    "ami-x-pswl": "阿美_海岸",
    "ami-x-frng": "阿美_馬蘭",
    "ami-x-pld": "阿美_恆春",
    "tay-x-sql": "泰雅_賽考利克",
    "tay-x-sul": "泰雅_澤敖利",
    "tay-x-cql": "泰雅_四季",
    "tay-x-kls": "泰雅_宜蘭澤敖利",
    "tay-x-mtuw": "泰雅_汶水",
    "tay-x-plngw": "泰雅_萬大",
    "pwn-x-kcdsn": "排灣_東",
    "pwn-x-vnrn": "排灣_北",
    "pwn-x-pnvn": "排灣_中",
    "pwn-x-ynvl": "排灣_南",
    "bnn-x-td": "布農_卓群",
    "bnn-x-bkh": "布農_卡群",
    "bnn-x-vtn": "布農_丹群",
    "bnn-x-bnz": "布農_巒群",
    "bnn-x-isbk": "布農_郡群",
    "pyu-x-pym": "卑南_南王",
    "pyu-x-ktrp": "卑南_知本",
    "pyu-x-mkzy": "卑南_西群",
    "pyu-x-ksvk": "卑南_建和",
    "dru-x-trmk": "魯凱_東",
    "dru-x-ngdr": "魯凱_霧台",
    "dru-x-lbw": "魯凱_大武",
    "dru-x-kgdv": "魯凱_多納",
    "dru-x-tldr": "魯凱_茂林",
    "dru-x-opnh": "魯凱_萬山",
    "tsu": "鄒",
    "xsy": "賽夏",
    "tao": "雅美",
    "ssf": "邵",
    "ckv": "噶瑪蘭",
    "trv-x-truku": "太魯閣",
    "szy": "撒奇萊雅",
    "trv-x-td": "賽德克_都達",
    "trv-x-tgdy": "賽德克_德固達雅",
    "trv-x-trk": "賽德克_德鹿谷",
    "sxr": "拉阿魯哇",
    "xnb": "卡那卡那富",
}
