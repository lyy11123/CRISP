# 生成副本的模型变更

## decode/ln/A3h
```diff
```
## decode/ln/Gemini
```diff
```
## decode/mha/A3h
```diff
--- native formatted
+++ H/I generated
@@ -23,6 +23,7 @@
 from add_communication import add_beha_producers, build_event, add_mediumdata, add_broadcast
 from data_notation import tensor_notation
 from conv1d import Conv1d
+from hi_config import shared_history_conv1d
 from transpose import Transpose
 from elerope import Elerope
 from matmul import Matmul
@@ -202,7 +203,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
@@ -230,7 +231,7 @@
     )
     qrope_data_tag = qrope_oper.target_data_tags[0]
 
-    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = Conv1d(
+    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_qkt",
@@ -258,7 +259,7 @@
     )
     s_data_tag = s_oper.target_data_tags[0]        
 
-    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = Conv1d(
+    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_p",
```
## decode/mha/Gemini
```diff
--- native formatted
+++ H/I generated
@@ -23,6 +23,7 @@
 from add_communication import add_beha_producers, build_event, add_mediumdata, add_broadcast
 from data_notation import tensor_notation
 from conv1d import Conv1d
+from hi_config import shared_history_conv1d
 from transpose import Transpose
 from elerope import Elerope
 from matmul import Matmul
@@ -202,7 +203,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
@@ -230,7 +231,7 @@
     )
     qrope_data_tag = qrope_oper.target_data_tags[0]
 
-    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = Conv1d(
+    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_qkt",
@@ -258,7 +259,7 @@
     )
     s_data_tag = s_oper.target_data_tags[0]        
 
-    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = Conv1d(
+    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_p",
```
## decode/proj/A3h
```diff
```
## decode/proj/Gemini
```diff
```
## decode/ffn/A3h
```diff
--- native formatted
+++ H/I generated
@@ -187,7 +187,7 @@
     data_dict=data_dict,
     beha_dict=beha_dict,
     oper_name="MLP"+"mlp2",
-    source_data_tags=[activation_data_tag],
+    source_data_tags=[ln_data_tag],
     oper_split_list=mlp2_parallelisms_list,
     oper_tag=act_next_oper_tag,
     online_data_tag=next_online_data_tag,
```
## decode/ffn/Gemini
```diff
--- native formatted
+++ H/I generated
@@ -187,7 +187,7 @@
     data_dict=data_dict,
     beha_dict=beha_dict,
     oper_name="MLP"+"mlp2",
-    source_data_tags=[activation_data_tag],
+    source_data_tags=[ln_data_tag],
     oper_split_list=mlp2_parallelisms_list,
     oper_tag=act_next_oper_tag,
     online_data_tag=next_online_data_tag,
```
## prefill/ln/A3h
```diff
```
## prefill/ln/Gemini
```diff
```
## prefill/mha/A3h
```diff
--- native formatted
+++ H/I generated
@@ -202,7 +202,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
```
## prefill/mha/Gemini
```diff
--- native formatted
+++ H/I generated
@@ -202,7 +202,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
```
## prefill/proj/A3h
```diff
```
## prefill/proj/Gemini
```diff
```
## prefill/ffn/A3h
```diff
```
## prefill/ffn/Gemini
```diff
```
