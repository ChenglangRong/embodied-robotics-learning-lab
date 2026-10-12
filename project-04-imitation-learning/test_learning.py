import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from learning import MLP, TwoLinkReachEnv, encode, expert, collect, specifications, protocol, rollout, fit, aggregate


class LearningTests(unittest.TestCase):
    def test_invalid_collection_options_are_rejected(self):
        specs=specifications(1,5,"train")
        for options in ({"query_every":0},{"budget":0},{"query_every":5}):
            with self.assertRaises(ValueError):collect(specs,**options)

    def test_fit_preserves_frozen_normalization(self):
        obs=np.array([[i]*6 for i in range(8)],dtype=float)
        data={"obs":obs,"action":np.zeros((8,2))}
        validation={"obs":obs+100,"action":np.zeros((8,2))}
        mean,std=obs.mean(0),obs.std(0)
        model=MLP(0,width=4,mean=mean,std=std)
        trained,_,_=fit(model,data,validation,2,10,batch_size=4)
        np.testing.assert_array_equal(trained.mean,mean)
        np.testing.assert_array_equal(trained.std,std)

    def test_selected_checkpoint_uses_validation_loss(self):
        obs=np.ones((4,6))
        data={"obs":obs,"action":np.zeros((4,2))}
        trained,history,selected=fit(MLP(2,width=4),data,data,5,10,batch_size=4)
        best=min(history,key=lambda row:row["validation_mse"])
        self.assertEqual(selected,best["update"])
        self.assertAlmostEqual(trained.loss_grad(obs,data["action"])[0],best["validation_mse"])

    def test_observation_contract(self):
        env=TwoLinkReachEnv()
        env.reset(seed=11)
        obs=encode(env)
        self.assertEqual(obs.shape,(6,))
        self.assertAlmostEqual(obs[0]**2+obs[1]**2,1.)
        self.assertAlmostEqual(obs[2]**2+obs[3]**2,1.)
        np.testing.assert_allclose(obs[4:],env.goal-env.position())

    def test_expert_data_alignment_and_limits(self):
        data,_=collect(specifications(2,5,"train"))
        np.testing.assert_allclose(data["next_q"],data["pre_q"]+.02*data["executed_action"],atol=1e-12)
        np.testing.assert_array_equal(data["action"],data["executed_action"])
        self.assertLessEqual(np.abs(data["action"]).max(),2.)

    def test_dagger_labels_and_executed_actions_are_separate(self):
        zero=lambda env:np.zeros(2)
        data,episodes=collect(specifications(1,5,"dagger"),behavior=zero,query_every=5,budget=12)
        self.assertEqual(len(data["obs"]),12)
        self.assertTrue(episodes[0]["budget_truncated"])
        self.assertGreater(np.abs(data["action"]).max(),0.)
        np.testing.assert_array_equal(data["executed_action"],0.)
        np.testing.assert_array_equal(data["next_q"],data["pre_q"])
        np.testing.assert_array_equal(data["step"],np.arange(12)*5)
        env=TwoLinkReachEnv()
        for q,goal,obs,label in zip(data["pre_q"],data["goal"],data["obs"],data["action"]):
            env.q=q.copy()
            env.goal=goal.copy()
            np.testing.assert_allclose(encode(env),obs)
            np.testing.assert_allclose(expert(env),label)

    def test_protocol_episode_and_reset_splits(self):
        specs=protocol()
        sets=[{x["id"] for x in specs[k]} for k in ("train","validation","test","ood")]
        for i,a in enumerate(sets):
            for b in sets[i+1:]:self.assertFalse(a&b)
        seeds=[x["reset_seed"] for k in ("train","validation","test","ood") for x in specs[k]]
        self.assertEqual(len(seeds),len(set(seeds)))

    def test_ood_domain_is_frozen_separate(self):
        specs=protocol()
        for split in ("train","validation","test","ood"):
            for s in specs[split]:
                angle=np.arctan2(s["goal"][1],s["goal"][0])
                self.assertTrue(-1.3<=angle<=-.1 if split=="ood" else .1<=angle<=1.3)

    def test_backprop_matches_finite_differences(self):
        rng=np.random.default_rng(4)
        model=MLP(0,width=4)
        obs=rng.normal(size=(5,6))
        target=rng.normal(size=(5,2))
        _,grads=model.loss_grad(obs,target)
        for key,value in model.params.items():
            for index in (tuple(0 for _ in value.shape),tuple(n-1 for n in value.shape)):
                old=value[index]
                value[index]=old+1e-6
                plus=model.loss_grad(obs,target)[0]
                value[index]=old-1e-6
                minus=model.loss_grad(obs,target)[0]
                value[index]=old
                self.assertAlmostEqual((plus-minus)/2e-6,grads[key][index],places=5)

    def test_adam_reduces_simple_supervised_loss(self):
        rng=np.random.default_rng(9)
        obs=rng.normal(size=(50,6))
        targets=obs[:,:2]*.3
        model=MLP(0,width=12)
        initial=model.loss_grad(obs,targets)[0]
        for _ in range(200):
            _,grad=model.loss_grad(obs,targets)
            model.update(grad)
        self.assertLess(model.loss_grad(obs,targets)[0],initial*.15)

    def test_model_save_load_preserves_predictions(self):
        model=MLP(3,width=6,mean=np.arange(6)*.1,std=np.ones(6)*.8)
        obs=np.ones(6)*.2
        with tempfile.TemporaryDirectory(prefix="bc-test-") as directory:
            file=Path(directory)/"model.npz"
            model.save(file)
            restored=MLP.load(file)
            np.testing.assert_array_equal(restored.forward(obs)[0],model.forward(obs)[0])

    def test_clones_preserve_optimizer_state(self):
        model=MLP(0,width=4)
        _,grad=model.loss_grad(np.ones((3,6)),np.zeros((3,2)))
        model.update(grad)
        clone=copy.deepcopy(model)
        self.assertEqual(clone.t,1)
        model.update(grad)
        clone.update(grad)
        for key in model.params:np.testing.assert_array_equal(model.params[key],clone.params[key])

    def test_expert_and_zero_share_success_rule(self):
        spec={"id":"test","split":"test","reset_seed":11,"goal":[.55,.55]}
        successful,_=rollout(spec,expert)
        failed,trace=rollout(spec,lambda env:np.zeros(2))
        self.assertTrue(successful["success"])
        self.assertLessEqual(successful["final_error_m"],.001)
        self.assertEqual(failed["termination"],"timeout")
        self.assertEqual(len(trace),600)

    def test_budget_exactly_matched(self):
        data,episodes=collect(specifications(12,3,"fresh"),budget=153)
        self.assertEqual(len(data["obs"]),153)
        self.assertEqual(sum(e["labels"] for e in episodes),153)

    def test_aggregate_keeps_episode_ids(self):
        a,_=collect(specifications(1,3,"train"),budget=10)
        b,_=collect(specifications(1,4,"dagger"),budget=8)
        result=aggregate(a,b)
        self.assertEqual(len(result["obs"]),18)
        self.assertEqual(set(result["episode_id"]),{"train-0000","dagger-0000"})


if __name__=="__main__":
    unittest.main()
