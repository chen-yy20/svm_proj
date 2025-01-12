
import numpy as np
import matplotlib.pyplot as plt
train_acc=[0.9325756933254177, 0.9419048426766178, 0.9479263845305742, 0.9484352472224578, 0.9503858875413451, 0.9546264099737087, 0.9516580442710542, 0.9564922398439488, 0.9572555338817742, 0.958103638368247, 0.9587821219574252, 0.9579340174709524, 0.9597998473411924, 0.9592061742006616, 0.955304893562887, 0.9612416249681961, 0.9601390891357815, 0.9587821219574252, 0.9591213637520143, 0.9575947756763633, 0.9596302264438978, 0.9608175727249597, 0.9607327622763124, 0.9583580697141888, 0.9608175727249597, 0.9597150368925451, 0.9625985921465524, 0.964634042914087, 0.962768213043847, 0.9641251802222034, 0.9614112458654906, 0.9609871936222543, 0.9629378339411415, 0.9651429056059706, 0.960902383173607, 0.9634466966330252, 0.9536086845899415, 0.9595454159952506, 0.9602238995844288, 0.9525061487575269, 0.9624289712492579, 0.9623441608006106, 0.9625137816979051, 0.9631074548384361, 0.9624289712492579, 0.9659061996437961, 0.9652277160546179, 0.9645492324654398, 0.9650580951573234, 0.960902383173607, 0.9632770757357306, 0.9601390891357815, 0.9570011025358324, 0.9619201085573743, 0.9637859384276143, 0.963701127978967, 0.9622593503519634, 0.9673479772707998, 0.9632770757357306, 0.9259604783309304, 0.9609871936222543, 0.9645492324654398, 0.9619201085573743, 0.9629378339411415, 0.9474175218386905, 0.9550504622169451, 0.9631074548384361, 0.9657365787465015, 0.9542023577304724, 0.958103638368247, 0.9671783563735052, 0.9628530234924942, 0.9610720040709015, 0.9660758205410906, 0.9592909846493088, 0.9661606309897379, 0.963701127978967, 0.9675175981680944, 0.9532694427953524, 0.9371554575523704, 0.9658213891951488, 0.9659061996437961, 0.9621745399033161, 0.9541175472818251, 0.9551352726655924, 0.9667543041302689, 0.9620897294546689, 0.960902383173607, 0.9656517682978543, 0.9390212874226105, 0.9661606309897379, 0.9634466966330252, 0.9593757950979561, 0.9653125265032652, 0.960902383173607, 0.964634042914087, 0.9276566873038758, 0.9476719531846324, 0.9628530234924942]
test_acc=[0.9362129583124058, 0.9427423405323958, 0.9467604218985435, 0.9472626820693119, 0.9502762430939227, 0.951783023606228, 0.951783023606228, 0.9552988448016072, 0.9547965846308388, 0.9578101456554495, 0.958312405826218, 0.958312405826218, 0.9598191863385234, 0.957307885484681, 0.9568056253139126, 0.9603214465092919, 0.9618282270215972, 0.9623304871923657, 0.9588146659969864, 0.958312405826218, 0.9633350075339026, 0.9578101456554495, 0.9618282270215972, 0.958312405826218, 0.9633350075339026, 0.957307885484681, 0.9608237066800602, 0.9613259668508287, 0.9633350075339026, 0.9633350075339026, 0.9623304871923657, 0.9643395278754395, 0.963837267704671, 0.9618282270215972, 0.9623304871923657, 0.9643395278754395, 0.9477649422400803, 0.9628327473631341, 0.9633350075339026, 0.9457559015570065, 0.9603214465092919, 0.958312405826218, 0.9643395278754395, 0.9663485685585133, 0.9618282270215972, 0.9643395278754395, 0.9663485685585133, 0.9643395278754395, 0.9648417880462079, 0.9628327473631341, 0.9658463083877449, 0.9558011049723757, 0.9467604218985435, 0.9578101456554495, 0.9648417880462079, 0.9618282270215972, 0.9628327473631341, 0.9648417880462079, 0.9608237066800602, 0.9171270718232044, 0.9628327473631341, 0.963837267704671, 0.9623304871923657, 0.9633350075339026, 0.9477649422400803, 0.9532898041185334, 0.9618282270215972, 0.9653440482169764, 0.9568056253139126, 0.957307885484681, 0.963837267704671, 0.9668508287292817, 0.9532898041185334, 0.9663485685585133, 0.9608237066800602, 0.9653440482169764, 0.957307885484681, 0.9658463083877449, 0.9512807634354595, 0.9362129583124058, 0.9628327473631341, 0.9673530889000502, 0.958312405826218, 0.9537920642893019, 0.958312405826218, 0.9653440482169764, 0.9628327473631341, 0.9628327473631341, 0.9593169261677549, 0.9432446007031643, 0.9688598694123556, 0.9578101456554495, 0.9633350075339026, 0.9653440482169764, 0.9633350075339026, 0.9628327473631341, 0.9306880964339528, 0.939728779507785, 0.9623304871923657]
train_acc2=[0.9657365787465015, 0.9664150623356798, 0.9670087354762107, 0.9584428801628361, 0.9594606055466034, 0.9643796115681452, 0.9665846832329743, 0.9549656517682978, 0.9571707234331269, 0.9240098380120431, 0.9610720040709015, 0.9555593249088288, 0.9587821219574252, 0.9636163175303197, 0.9600542786871342, 0.9543719786277669, 0.9648884742600289, 0.9638707488762616, 0.9653973369519124, 0.9603935204817233, 0.9643796115681452, 0.961835298108727, 0.9518276651683487, 0.9614960563141379, 0.9253668051903995, 0.9561529980493597, 0.9592909846493088, 0.9469934695954542, 0.9584428801628361, 0.9571707234331269, 0.9124756169960139, 0.9247731320498686, 0.9374946993469595, 0.9580188279195997, 0.9503858875413451, 0.9585276906114834, 0.951912475616996, 0.9624289712492579, 0.9609871936222543, 0.9567466711898905, 0.9547960308710033, 0.9625137816979051, 0.9551352726655924, 0.9496225935035196, 0.9587821219574252, 0.9606479518276652, 0.8955983377152065, 0.9609871936222543, 0.9630226443897888, 0.9514036129251123, 0.9545415995250615, 0.9229921126282759, 0.9432618098549741, 0.9164617080824358, 0.9579340174709524, 0.951318802476465, 0.9525061487575269, 0.9581884488168942, 0.9564074293953015, 0.959969468238487, 0.9505555084386397, 0.9185819692986176, 0.9636163175303197, 0.9610720040709015, 0.9615808667627852, 0.9501314561954033, 0.9338478500551268, 0.9434314307522687, 0.9324060724281231, 0.9631074548384361, 0.9610720040709015, 0.9564074293953015, 0.950640318887287, 0.9587821219574252, 0.9003477228394539, 0.9505555084386397, 0.9569162920871851, 0.9335086082605377, 0.9570011025358324, 0.9611568145195488, 0.9436858620982105, 0.9617504876600798, 0.965566957849207, 0.9621745399033161, 0.9580188279195997, 0.9633618861843779, 0.9515732338224069, 0.9441947247900941, 0.9629378339411415, 0.9391060978712578]
test_acc2=[0.9678553490708187, 0.963837267704671, 0.9648417880462079, 0.9598191863385234, 0.9568056253139126, 0.9648417880462079, 0.9663485685585133, 0.9593169261677549, 0.957307885484681, 0.928679055750879, 0.9608237066800602, 0.9437468608739327, 0.9603214465092919, 0.9628327473631341, 0.9598191863385234, 0.9507785032646912, 0.9613259668508287, 0.9658463083877449, 0.9653440482169764, 0.9633350075339026, 0.9643395278754395, 0.9643395278754395, 0.9507785032646912, 0.9618282270215972, 0.928679055750879, 0.9568056253139126, 0.9618282270215972, 0.9558011049723757, 0.9537920642893019, 0.9578101456554495, 0.9085886489201407, 0.9291813159216474, 0.9326971371170266, 0.9542943244600703, 0.9507785032646912, 0.9522852837769965, 0.9532898041185334, 0.963837267704671, 0.9507785032646912, 0.957307885484681, 0.9507785032646912, 0.9683576092415871, 0.9558011049723757, 0.9477649422400803, 0.9593169261677549, 0.9552988448016072, 0.8884982420894023, 0.9532898041185334, 0.9623304871923657, 0.9558011049723757, 0.9532898041185334, 0.9246609743847313, 0.9427423405323958, 0.9181315921647414, 0.9568056253139126, 0.9492717227523857, 0.9457559015570065, 0.9563033651431442, 0.9542943244600703, 0.9563033651431442, 0.9492717227523857, 0.9161225514816675, 0.9623304871923657, 0.9588146659969864, 0.9613259668508287, 0.9487694625816173, 0.9216474133601206, 0.9392265193370166, 0.9271722752385736, 0.9588146659969864, 0.9598191863385234, 0.9593169261677549, 0.9563033651431442, 0.9608237066800602, 0.910095429432446, 0.9522852837769965, 0.9552988448016072, 0.9331993972877951, 0.9542943244600703, 0.9563033651431442, 0.9412355600200905, 0.9608237066800602, 0.9653440482169764, 0.9593169261677549, 0.9497739829231542, 0.9658463083877449, 0.9512807634354595, 0.9402310396785535, 0.9608237066800602, 0.9321948769462581]
train=train_acc+train_acc2
test=test_acc+test_acc2
C_values = np.concatenate((np.arange(1, 100), np.arange(101, 1000, 10)))  # 正则化系数C从1到99的整数


# 创建图形
plt.figure(figsize=(8, 6))

# 绘制训练准确率曲线
plt.plot(C_values, train, label='Training Accuracy', color='blue', linewidth=2)

# 绘制测试准确率曲线
plt.plot(C_values, test, label='Testing Accuracy', color='red', linewidth=2)

# 设置坐标轴标签
plt.xlabel('Regularization Coefficient C', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# 设置标题
plt.title('Training and Testing Accuracy vs. Regularization Coefficient C', fontsize=14)

# 添加网格
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()