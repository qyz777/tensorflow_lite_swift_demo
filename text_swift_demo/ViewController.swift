//
//  ViewController.swift
//  text_swift_demo
//
//  Created by Q YiZhong on 2019/7/7.
//  Copyright © 2019 YiZhong Qi. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        
        let text = "麦基砍28+18+5却充满寂寞 纪录之夜他的痛阿联最懂新浪体育讯上天对每个人都是公平的，贾维尔-麦基也不例外。今天华盛顿奇才客场104-114负于金州勇士，麦基就好不容易等到“捏软柿子”的机会，上半场打出现象级表现，只可惜无法一以贯之。最终，麦基12投9中，得到生涯最高的28分，以及平生涯最佳的18个篮板，另有5次封盖。此外，他11次罚球命中10个，这两项也均为生涯最高。如果在赛前搞个竞猜，上半场谁会是奇才阵中罚球次数最多的球员？若有人答曰“麦基”，不是恶搞就是脑残。但半场结束，麦基竟砍下22分(第二节砍下14分)。更罕见的，则是他仅出手7次，罚球倒是有11次，并命中了其中的10次。此外，他还抢下11个篮板，和勇士首发五虎总篮板数持平；他还送出3次盖帽，竟然比勇士全队上半场的盖帽总数还多1次！麦基为奇才带来了什么？除了得分方面异军突起，在罚球线上杀伤对手，率队紧咬住比分，用封盖威慑对手外，他在篮板上的贡献最为关键。众所周知，篮板就是勇士的生命线。3月3日的那次交锋前，时任代理主帅的兰迪-惠特曼在赛前甚至给沃尔和尼克-杨二人下达“篮板不少于10个”的硬性指标。惠特曼没疯，他深知守住了篮板阵地，就如同扼住了勇士的咽喉。上次交锋拿下16个篮板的大卫-李就说：“称霸篮板我们取胜希望就大些。我投中多少球无所谓，但我一定要保护篮板”。最终，勇士总篮板数以54-40领先。但今天，半场结束麦基却让李仅有5个篮板进账。造成这种局面的关键因素是身高。在2.11米的安德里斯-比德林斯伤停后，勇士内线也更为迷你。李2.06米，弗拉迪米尔-拉德马诺维奇2.08米，艾派-乌杜2.08米，路易斯-阿蒙德森2.06米。由此，2.13米又弹跳出众的麦基也就有些鹤立鸡群了。翻开本赛季中锋篮板效率榜，比德林斯位居第13位，麦基第20，李则是第31。可惜，麦基出彩不但超出了勇士预期，也超出了奇才预期，注定不可长久。第三节李砍下12分，全场26投15中砍下33分12个篮板5次助攻，麦基的防守不利则被放大。2分11秒，奇才失误，蒙塔-埃利斯带球直冲篮下，面对麦基的防守，他华丽的篮下360度转身上篮命中。全场掌声雷动下，麦基的身影却无比落寞。下半场麦基有多困顿？篮板被对方追上，全场勇士篮板仅落后2个；上半场拉风的罚球，在下半场竟然一次也没有。和阿联此役先扬后抑的表现如出一辙，麦基也吃尽了奇才内线缺兵少将的苦头。(魑魅)"
        
        TextClassifier.shared.loadInfo {
            TextClassifier.shared.runModel(with: text) { (res) in
                print(res)
            }
        }
    }


}
